import os
import json
import math
import tempfile
from typing import List, Dict, Any
import gc
import psutil
import logging
import argparse

import numpy as np
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
import librosa
import soundfile as sf
import sys
import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# Add parent directory to path to import custom dataloader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import VideoQADaloader

# Set environment variables
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_video_chunk_content(video_path, flatten=False):
    """
    Extract video frames and audio content from video file.
    Ensures audio index consistency across all segments.
    
    Args:
        video_path: Path to video file
        flatten: Whether to flatten the output list
        
    Returns:
        List of video chunks containing frames and audio segments
    """
    video = VideoFileClip(video_path)
    logging.info(f'Video duration: {video.duration}s')
    
    # Check if video has audio track
    if video.audio is None:
        logging.warning(f"Warning: Video {video_path} has no audio track")
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
            logging.error(f"Audio extraction failed: {e}, using silent audio")
            duration = video.duration
            sr = 16000
            audio_np = np.zeros(int(duration * sr), dtype=np.float32)
    
    # Ensure audio length matches video duration
    expected_audio_length = int(video.duration * sr)
    if len(audio_np) != expected_audio_length:
        logging.info(f"Adjusting audio length: {len(audio_np)} -> {expected_audio_length}")
        if len(audio_np) < expected_audio_length:
            audio_np = np.pad(audio_np, (0, expected_audio_length - len(audio_np)), mode='constant', constant_values=0)
        else:
            audio_np = audio_np[:expected_audio_length]
    
    num_units = math.ceil(video.duration)
    
    # Pre-compute all valid audio segments to ensure index consistency
    valid_segments = []
    for i in range(num_units):
        start_idx = sr * i
        end_idx = sr * (i + 1)
        
        # Check boundary conditions
        if start_idx >= len(audio_np):
            logging.warning(f"Warning: Segment {i} start index out of range, skipping")
            continue
            
        if end_idx > len(audio_np):
            available_audio = audio_np[start_idx:]
            if len(available_audio) < sr * 0.1:  # Skip if remaining audio is too short
                logging.warning(f"Warning: Segment {i} remaining audio too short ({len(available_audio)} samples), skipping")
                continue
            # Adjust end_idx to actual audio length for last segment
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
            
            # Ensure audio length is 1 second (16000 samples), except for last segment
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
            logging.error(f"Error processing segment {segment_idx}: {e}")
            video.close()
            raise e
    
    logging.info(f"Processed {len(valid_segments)} valid audio segments out of {num_units} total")
    
    video.close()
    return contents


def write_result_to_file(result, output_file):
    """Write a single result to JSON file"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Create empty list if file doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
    
    # Read existing results
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        existing_results = []
    
    # Append new result
    existing_results.append(result)
    
    # Write back to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)


def evaluate_model(dataloader, model, processor, tokenizer, output_file, max_duration=None):
    """
    Evaluate model on video QA dataset with real-time result writing.
    
    Args:
        dataloader: VideoQADaloader instance
        model: VideoLLaMA2 model
        processor: Video/audio processor
        tokenizer: Text tokenizer
        output_file: Path to save results
        max_duration: Maximum video duration to process (in seconds)
    """
    qa_pairs = dataloader.get_all_qa_pairs()
    
    # Filter by duration if specified
    if max_duration is not None:
        qa_pairs = [qa for qa in qa_pairs if qa.get('duration', 0) <= max_duration]
        logging.info(f"Filtered QA pairs by duration <= {max_duration}s: {len(qa_pairs)} pairs")
    
    # Memory monitoring
    logging.info(f"Starting evaluation, initial memory usage: {psutil.virtual_memory().percent}%")
    logging.info(f"Total QA pairs: {len(qa_pairs)}")
    torch.cuda.empty_cache()
    gc.collect()

    processed_count = 0
    error_count = 0
    correct_count = 0

    for idx, item in enumerate(qa_pairs):
        video_path = item['video_path']
        video_name = os.path.basename(video_path)
        
        logging.info(f"\nProcessing {idx+1}/{len(qa_pairs)}: {video_name}")
        
        # Initialize result object
        result = {
            "video": video_name,
            "duration": item.get("duration", 0),
            "question": item.get("question", ""),
            "options": item.get("options", []),
            "answer": item.get("answer", ""),
            "model_answer": "",
            "is_correct": False
        }

        # Check if video exists
        if not os.path.exists(video_path):
            logging.error(f"Video not found: {video_path}")
            result["model_answer"] = f"Error: Video file not found - {video_path}"
            write_result_to_file(result, output_file)
            error_count += 1
            continue

        try:
            # Process video with VideoLLaMA2 processor
            video_tensor = processor['video'](video_path)

            # Build prompt
            question = item.get("question")
            options = item.get("options", [])
            correct_answer = item.get("answer")
            options_text = "\n".join(options)

            instruct = (
                "You are given a video. Based on the content of the video, answer the following question:\n\n"
                f"Question:\n{question}\n\n"
                f"Options:\n{options_text}\n\n"
                "Answer with the option's letter directly (e.g., A, B, C, or D). "
                "If your access to the video content is limited, at least one option that is more likely than the others must be chosen. "
                "Do not give any other reason for not choosing!"
            )

            # Model inference
            try:
                output = mm_infer(
                    image_or_video=video_tensor,
                    instruct=instruct,
                    model=model,
                    tokenizer=tokenizer,
                    do_sample=False,
                )
                
                model_answer = output.strip()
                is_correct = (model_answer.upper() == str(correct_answer).strip().upper())

                result["model_answer"] = model_answer
                result["is_correct"] = is_correct

                if is_correct:
                    correct_count += 1

                processed_count += 1
                
            except torch.cuda.OutOfMemoryError as oom_error:
                error_msg = f"Error: CUDA out of memory - {str(oom_error)}"
                logging.error(f"CUDA OOM: {oom_error}")
                result["model_answer"] = error_msg
                result["is_correct"] = False
                error_count += 1
                
                # Clear memory and continue
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as model_error:
                error_msg = f"Error: {type(model_error).__name__} - {str(model_error)}"
                logging.error(f"Model inference error: {model_error}")
                result["model_answer"] = error_msg
                result["is_correct"] = False
                error_count += 1

        except Exception as e:
            error_msg = f"Error: {type(e).__name__} - {str(e)}"
            logging.error(f"Error processing video {video_name}: {e}")
            result["model_answer"] = error_msg
            result["is_correct"] = False
            error_count += 1
        
        # Write result to file
        write_result_to_file(result, output_file)
        logging.info(f"Result written: {video_name} - {result['model_answer'][:50]}...")
        
        # Periodic memory cleanup
        if (idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            memory_percent = psutil.virtual_memory().percent
            logging.info(f"Processed {idx+1} videos, current memory usage: {memory_percent}%")
            
            # Force cleanup if memory usage is too high
            if memory_percent > 85:
                logging.warning("High memory usage detected, forcing cleanup...")
                torch.cuda.empty_cache()
                gc.collect()
    
    # Final statistics
    logging.info(f"\nEvaluation complete!")
    logging.info(f"Successfully processed: {processed_count} videos")
    logging.info(f"Errors: {error_count}")
    if processed_count > 0:
        logging.info(f"Accuracy: {(correct_count / processed_count) * 100:.2f}% ({correct_count}/{processed_count})")


def main():
    parser = argparse.ArgumentParser(description='VideoLLaMA2 Evaluation Script')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to VideoLLaMA2 model')
    parser.add_argument('--input_file', type=str, default='data.json',
                       help='Path to input QA JSON file')
    parser.add_argument('--video_dir', type=str, default='./videos',
                       help='Directory containing video files')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save results (default: {model_name}_results.json)')
    parser.add_argument('--max_duration', type=int, default=None,
                       help='Maximum video duration to process (in seconds)')
    parser.add_argument('--log_file', type=str, default='videollama2_eval.log',
                       help='Path to log file')
    parser.add_argument('--cuda_device', type=str, default='0',
                       help='CUDA device to use')
    
    args = parser.parse_args()
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set random seed
    set_seed(42)
    
    # Determine output file name
    if args.output_file is None:
        model_basename = os.path.basename(args.model_path)
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = f"{model_basename}_{input_basename}_results.json"
    
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Video directory: {args.video_dir}")
    logging.info(f"Output file: {args.output_file}")
    logging.info(f"Max duration: {args.max_duration}s" if args.max_duration else "Max duration: No limit")
    
    # Load dataloader
    dataloader = VideoQADaloader(args.input_file, args.video_dir)
    
    # Initialize model
    logging.info("Initializing model...")
    disable_torch_init()
    model, processor, tokenizer = model_init(model_path=args.model_path)
    model = model.eval().cuda()
    logging.info("Model loaded successfully")
    
    # Run evaluation
    evaluate_model(dataloader, model, processor, tokenizer, args.output_file, args.max_duration)
    
    logging.info(f"\nEvaluation complete! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()