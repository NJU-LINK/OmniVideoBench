import os
import json
import math
import tempfile
import argparse
from typing import List, Dict, Any
import gc
import psutil

import numpy as np
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
import librosa
import soundfile as sf
import sys
import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer

from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# Add project root to path for relative imports
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from dataloader import VideoQADaloader
from utils.utils import set_seed, filter_qa_pairs_by_duration

set_seed(42)


# Add problematic videos to blacklist
PROBLEMATIC_VIDEOS = set()

def get_video_chunk_content(video_path, flatten=False):
    """Improved video content extraction function ensuring audio index consistency"""
    video = VideoFileClip(video_path)
    print('video_duration:', video.duration)
    
    # Check if the video has an audio track
    if video.audio is None:
        print(f"Warning: Video {video_path} has no audio track")
        duration = video.duration
        sr = 16000
        audio_np = np.zeros(int(duration * sr), dtype=np.float32)
    else:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                temp_audio_file_path = temp_audio_file.name
                video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
                audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
        except Exception as e:
            print(f"Audio extraction failed: {e}, using silent audio")
            duration = video.duration
            sr = 16000
            audio_np = np.zeros(int(duration * sr), dtype=np.float32)
    
    # Ensure audio length matches video duration
    expected_audio_length = int(video.duration * sr)
    if len(audio_np) != expected_audio_length:
        print(f"Audio length adjustment: {len(audio_np)} -> {expected_audio_length}")
        if len(audio_np) < expected_audio_length:
            audio_np = np.pad(audio_np, (0, expected_audio_length - len(audio_np)), mode='constant', constant_values=0)
        else:
            audio_np = audio_np[:expected_audio_length]
    
    num_units = math.ceil(video.duration)
    
    # Pre-calculate all valid audio segments to ensure index consistency
    valid_segments = []
    for i in range(num_units):
        start_idx = sr * i
        end_idx = sr * (i + 1)
        
        # Check boundary conditions
        if start_idx >= len(audio_np):
            print(f"Warning: Audio segment {i} start index out of range, skipping")
            continue
            
        if end_idx > len(audio_np):
            available_audio = audio_np[start_idx:]
            if len(available_audio) < sr * 0.1:  # If remaining audio is too short, skip
                print(f"Warning: Audio segment {i} remaining audio too short ({len(available_audio)} samples), skipping")
                continue
            # For the last segment, adjust end_idx to actual audio length
            end_idx = len(audio_np)
        
        valid_segments.append((i, start_idx, end_idx))
    
    # Generate content based on valid segments
    contents = []
    for segment_idx, start_idx, end_idx in valid_segments:
        try:
            # Get frame
            frame_time = min(segment_idx + 1, video.duration - 0.001)
            frame = video.get_frame(frame_time)
            image = Image.fromarray((frame).astype(np.uint8))
            
            # Extract audio segment
            audio = audio_np[start_idx:end_idx]
            
            # Ensure audio length is 1 second (16000 samples), except for the last segment
            if len(audio) < sr and segment_idx < num_units - 1:
                audio = np.pad(audio, (0, sr - len(audio)), mode='constant', constant_values=0)
            elif len(audio) > sr:
                audio = audio[:sr]
            
            # Ensure data type
            audio = audio.astype(np.float32)
            
            if flatten:
                contents.extend(["<unit>", image, audio])
            else:
                contents.append(["<unit>", image, audio])
                
        except Exception as e:
            print(f"Error processing segment {segment_idx}: {e}")
            # If a single segment fails, mark the entire video as problematic
            video.close()
            raise e
    
    print(f"Processed {len(valid_segments)} valid audio segments out of {num_units} total segments")
    
    video.close()
    return contents

def write_result_to_file(result, output_file):
    """Write single result to JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # If file doesn't exist, create empty list
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
    
    # Read existing results
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        existing_results = []
    
    # Add new result
    existing_results.append(result)
    
    # Write back to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

def evaluate_model(dataloader, model, tokenizer, output_file,max_duration):
    """Improved model evaluation function with real-time result writing"""

    # Use filter_qa_pairs_by_duration to filter QA pairs
    qa_pairs = filter_qa_pairs_by_duration(dataloader.get_all_qa_pairs(),max_duration)
    
    # Memory monitoring
    print(f"Starting evaluation, initial memory usage: {psutil.virtual_memory().percent}%")
    print(f"Filtered QA pairs count: {len(qa_pairs)}")
    torch.cuda.empty_cache()
    gc.collect()

    processed_count = 0
    error_count = 0

    # Number of correctly answered questions
    right_count = 0

    for idx, item in enumerate(qa_pairs):
        video_path = item['video_path']
        unique_id = item.get('unique_id', f"video_{idx}")
        video_name = os.path.basename(video_path)
        
        print(f"\nProcessing video {idx+1}/{len(qa_pairs)}: {video_name}")
        
        # Check if in blacklist
        if video_path in PROBLEMATIC_VIDEOS:
            print(f"Skipping problematic video: {video_name}")
            continue
            
        # Initialize result object
        result = {
            "unique_id": unique_id,
            "video": video_name,
            "duration": item.get("duration", 0),
            "question": item.get("question", ""),
            "options": item.get("options",[]),
            "correct_answer": item.get("answer", ""),
            "model_answer": "",
            "is_correct": False
        }

        # Process options format
        options = item.get("options", [])
        options_text = "\n".join(options)

        if not os.path.exists(video_path):
            print(f"Video does not exist: {video_path}")
            result["model_answer"] = f"Error: Video file does not exist - {video_path}"
            write_result_to_file(result, output_file)
            error_count += 1
            continue

        try:
            # --- HumanOmni data preprocessing ---
            # Use HumanOmni's processor to handle video and audio
            # processor['video'] and processor['audio'] directly accept file paths
            video_tensor = processor['video'](video_path)
            audio_tensor = processor['audio'](video_path)[0] # processor['audio'] returns (tensor, sample_rate)

            # --- Build Prompt ---
            question = item.get("question")
            options = item.get("options", [])
            correct_answer = item.get("answer")
            options_text = "\n".join(options)

            instruct = (
                "You are given a video. Based on the content of the video, answer the following question:\n\n"
                f"Question:\n{question}\n\n"
                f"Options:\n{options_text}\n\n"
                "Answer with the option's letter directly(e.g., A, B, C, or D)."
                "If your access to the video content is limited, at least one option that is more likely than the others must be chosen."
                "Mustn't give any other reason for can not choose!"
            )

            # ----Human Omni model inference
            try:
                output = mm_infer(
                    image_or_video=video_tensor,
                    instruct=instruct,
                    model=model,
                    tokenizer=tokenizer,
                    audio=audio_tensor,
                    modal='video_audio',
                    question=question, # mm_infer requires question parameter
                    bert_tokeni=bert_tokenizer,
                    do_sample=False,
                )
                
                model_answer = output.strip()
                is_correct = (model_answer.upper() == str(correct_answer).strip().upper())

                result["model_answer"] = model_answer
                result["is_correct"] = is_correct

                if is_correct == True:
                    right_count += 1

                processed_count += 1                
                
            except torch.cuda.OutOfMemoryError as oom_error:
                error_msg = f"Error: OutOfMemoryError - CUDA out of memory. {str(oom_error)}"
                print(f"CUDA out of memory: {oom_error}")
                result["model_answer"] = error_msg
                result["is_correct"] = False
                error_count += 1
                
                # Clean memory and continue
                torch.cuda.empty_cache()
                gc.collect()
                
            except AssertionError as ae:
                error_msg = f"Error: AssertionError - Audio index mismatch: {str(ae)}"
                print(f"Assertion error: {ae}")
                result["model_answer"] = error_msg
                result["is_correct"] = False
                PROBLEMATIC_VIDEOS.add(video_path)
                error_count += 1
                
            except Exception as model_error:
                error_msg = f"Error: {type(model_error).__name__} - {str(model_error)}"
                print(f"Model inference error: {model_error}")
                result["model_answer"] = error_msg
                result["is_correct"] = False
                PROBLEMATIC_VIDEOS.add(video_path)
                error_count += 1

        except Exception as e:
            error_msg = f"Error: {type(e).__name__} - {str(e)}"
            print(f"Error processing video {video_name}: {e}")
            result["model_answer"] = error_msg
            result["is_correct"] = False
            PROBLEMATIC_VIDEOS.add(video_path)
            error_count += 1
        
        # Write result to file
        write_result_to_file(result, output_file)
        print(f"Result written: {result['unique_id']} - {result['model_answer'][:50]}...")
        
        # Periodic memory cleanup
        if (idx + 1) % 5 == 0:  # More frequent cleanup
            torch.cuda.empty_cache()
            gc.collect()
            memory_percent = psutil.virtual_memory().percent
            print(f"Video {idx+1} processed, current memory usage: {memory_percent}%")
            
            # Force cleanup if memory usage is too high
            if memory_percent > 85:
                print("Memory usage too high, performing forced cleanup...")
                torch.cuda.empty_cache()
                gc.collect()
    
    print(f"\nEvaluation completed!")
    print(f"Successfully processed: {processed_count} videos")
    print(f"Error count: {error_count}")
    print(f"Problematic videos count: {len(PROBLEMATIC_VIDEOS)}")
    print(f"Accuracy: {( right_count / processed_count ) * 100: .2f}%")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="HumanOmni model evaluation script")
    
    # Data related arguments
    parser.add_argument("--task_name", type=str, default="qa_data",
                       help="Task name for output file naming")
    parser.add_argument("--data_json_file", type=str, required=True,
                       help="Path to QA data JSON file")
    parser.add_argument("--video_dir", type=str, required=True,
                       help="Path to video files directory")
    
    # Model related arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="Path to model directory")
    parser.add_argument("--bert_model", type=str, default="./bert-base-uncased",
                       help="Path to BERT model")
    
    # Output related arguments
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                       help="Output directory for results")
    
    # Evaluation parameters
    parser.add_argument("--max_duration", type=int, default=6000,
                       help="Maximum video duration in seconds")
    parser.add_argument("--cuda_visible_devices", type=str, default="0",
                       help="CUDA visible devices")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    
    # Build output file path
    model_basename = os.path.basename(args.model_name)
    output_file = os.path.join(args.output_dir, f"{model_basename}_{args.task_name}.json")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 50)
    print("HumanOmni Evaluation Configuration:")
    print(f"Task Name: {args.task_name}")
    print(f"Data File: {args.data_json_file}")
    print(f"Video Directory: {args.video_dir}")
    print(f"Model Path: {args.model_name}")
    print(f"BERT Model: {args.bert_model}")
    print(f"Output File: {output_file}")
    print(f"Max Duration: {args.max_duration} seconds")
    print(f"CUDA Devices: {args.cuda_visible_devices}")
    print("=" * 50)
    
    # Initialize data loader
    dataloader = VideoQADaloader(args.data_json_file, args.video_dir)

    # Initialize BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                        cache_dir=args.bert_model,
                                        local_files_only=True)
    print("BERT tokenizer initialized")
    
    # Disable torch initialization
    disable_torch_init()

    # Initialize model, processor and tokenizer
    model, processor, tokenizer = model_init(model_path=args.model_name)
    model = model.eval().cuda()
    print(f"Model device map: {model.hf_device_map}")

    # Execute evaluation
    evaluate_model(dataloader, model, tokenizer, output_file, args.max_duration)

    print(f"\nEvaluation completed, results saved to {output_file}")
