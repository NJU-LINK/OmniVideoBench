import os
import json
import math
import tempfile
from typing import List, Dict, Any
import gc
import psutil
import re

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, GenerationConfig
import random

import sys
# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from dataloader import VideoQADaloader

# Add Ming model path
ming_model_path = os.path.join(project_root, "model", "Ming")
sys.path.insert(0, ming_model_path)
from modeling_bailingmm import BailingMMNativeForConditionalGeneration

import torch.distributed as dist

# Set environment variables
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Blacklist for problematic videos
PROBLEMATIC_VIDEOS = set()

def extract_answer_letter(response):
    """
    Extract answer letter (A, B, C, D) from model response using regex
    """
    if not response:
        return ""
    
    # Clean response text
    response = response.strip()
    
    # Try multiple regex patterns to extract answer
    patterns = [
        r'^([ABCD])\.?\s*',  # Starts with A. or A
        r'([ABCD])\.?\s*$',  # Ends with A. or A
        r'answer\s*(is|:)?\s*([ABCD])',  # answer is A or answer: A
        r'choose\s*([ABCD])',  # choose A
        r'option\s*([ABCD])',  # option A
        r'([ABCD])\s*option',  # A option
        r'\b([ABCD])\b',  # single letter
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # If not found, return the first character of the response (if it's A-D)
    first_char = response[0].upper() if response else ""
    if first_char in ['A', 'B', 'C', 'D']:
        return first_char
    
    return ""

def generate_ming_response(messages, processor, model):
    """
    Generate response using Ming-omni model
    """
    try:
        # 1. Format inputs using chat template
        text = processor.apply_chat_template(messages, add_generation_prompt=True)

        # 2. Extract vision/audio data
        image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

        # 3. Prepare tensor inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        for k in inputs.keys():
            if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        # 4. Configure generation
        generation_config = GenerationConfig.from_dict({'no_repeat_ngram_size': 10})
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            use_cache=True,
            eos_token_id=processor.gen_terminator,
            generation_config=generation_config,
        )
        generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

        # 5. Decode output
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
        
    except Exception as e:
        print(f"Ming model inference error: {e}")
        raise e

def filter_qa_pairs_by_duration(qa_pairs, max_duration):
    """Filter QA pairs based on video duration."""
    filtered_qa_pairs = []
    for qa_pair in qa_pairs:
        duration = qa_pair.get("duration")
        if duration is not None and duration < max_duration:
            filtered_qa_pairs.append(qa_pair)
        elif duration is None:
            print(f"Warning: No duration data for video {qa_pair.get('video_path', 'Unknown')}, skipping.")
    
    print(f"Original dataset: {len(qa_pairs)} QA pairs")
    print(f"Filtered dataset (< {max_duration} seconds): {len(filtered_qa_pairs)} QA pairs")
    return filtered_qa_pairs

def write_result_to_file(result, output_file):
    """Write single result to JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
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
    
    # Add new result
    existing_results.append(result)
    
    # Write back to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

def evaluate_model(dataloader, model, processor, output_file, max_duration, max_frames=64):
    """Evaluate using Ming-omni model"""
    # Filter QA pairs by duration
    qa_pairs = filter_qa_pairs_by_duration(dataloader.get_all_qa_pairs(), max_duration)
    
    # Memory monitoring
    print(f"Starting evaluation, initial memory usage: {psutil.virtual_memory().percent}%")
    print(f"Number of filtered QA pairs: {len(qa_pairs)}")
    torch.cuda.empty_cache()
    gc.collect()

    processed_count = 0
    error_count = 0
    correct_count = 0

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
            "options": item.get("options", []),
            "correct_answer": item.get("answer", ""),
            "model_answer": "",
            "is_correct": False
        }

        # Process options format
        options = item.get("options", [])
        options_text = "\n".join(options)

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            result["model_answer"] = f"Error: Video file does not exist - {video_path}"
            write_result_to_file(result, output_file)
            error_count += 1
            continue

        try:
            question = item.get("question")
            correct_answer = item.get("answer")
            
            print(f"question: {question}")
            print(f"correct_answer: {correct_answer}")

            # Build Ming-omni message format
            question_prompt = (
                "You are given a video. Based on the content of the video, answer the following question:\n\n"
                f"Question:\n{question}\n\n"
                f"Options:\n{options_text}\n\n"
                "Answer with the option's letter directly(e.g., A, B, C, or D). "
                "If your access to the video content is limited, at least one option that is more likely than the others must be chosen. "
                "Mustn't give any other reason for can not choose!"
            )

            # Build Ming message format
            messages = [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "video", "video": video_path, "max_frames": max_frames, "sample": "uniform"},
                        {"type": "text", "text": question_prompt},
                    ],
                },
            ]

            # Model inference with error handling
            try:
                response = generate_ming_response(messages, processor, model)
                
                print(f"response: {response}")

                # Extract answer letter using regex
                extracted_answer = extract_answer_letter(response)
                model_answer = extracted_answer if extracted_answer else response.strip()
                
                print(f"extracted_answer: {extracted_answer}")
                
                is_correct = (extracted_answer.upper() == correct_answer.strip().upper()) if extracted_answer else False

                result["model_answer"] = model_answer
                result["is_correct"] = is_correct
                processed_count += 1
                
                # Count correct answers
                if is_correct:
                    correct_count += 1
                
            except torch.cuda.OutOfMemoryError as oom_error:
                error_msg = f"Error: OutOfMemoryError - CUDA out of memory. {str(oom_error)}"
                print(f"CUDA out of memory: {oom_error}")
                result["model_answer"] = error_msg
                result["is_correct"] = False
                error_count += 1
                
                # Clean memory and continue
                torch.cuda.empty_cache()
                gc.collect()
                
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
            print(f"Completed video {idx+1}, current memory usage: {memory_percent}%")
            
            # Force cleanup if memory usage is too high
            if memory_percent > 85:
                print("Memory usage too high, forcing cleanup...")
                torch.cuda.empty_cache()
                gc.collect()
    
    # Calculate accuracy
    accuracy = correct_count / processed_count if processed_count > 0 else 0.0
    
    print(f"\nEvaluation completed!")
    print(f"Successfully processed: {processed_count} videos")
    print(f"Correct answers: {correct_count}")
    print(f"Errors: {error_count}")
    print(f"Problematic videos: {len(PROBLEMATIC_VIDEOS)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        "total_processed": processed_count,
        "correct_answers": correct_count,
        "errors": error_count,
        "accuracy": accuracy,
        "problematic_videos": len(PROBLEMATIC_VIDEOS)
    }

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json_file", type=str, required=True, help="Path to the data JSON file")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--output_file", type=str, default="./eval_results/ming_omni_out.json", help="Output file path for evaluation results")
    parser.add_argument("--model_path", default="Ming-Lite-Omni-1.5",type=str, required=True, help="Path to the Ming-Lite-Omni model")
    parser.add_argument("--max_duration", type=int, default=6000, help="Maximum video duration in seconds")
    parser.add_argument("--max_frames", type=int, default=128, help="Maximum number of frames to extract from video")
    args = parser.parse_args()

    data_json_file = args.data_json_file
    video_dir = args.video_dir
    output_file = args.output_file
    model_path = args.model_path
    max_duration = args.max_duration
    max_frames = args.max_frames
    
    dataloader = VideoQADaloader(data_json_file, video_dir)
    
    # Load Ming-omni model
    print("Loading Ming-omni model...")
    
    # def init_distributed():
    #     if not dist.is_initialized():
    #         local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #         torch.cuda.set_device(local_rank)
    #         dist.init_process_group(backend="nccl")
    #     return int(os.environ.get("LOCAL_RANK", 0))

    # # Initialize distributed, then load model
    # local_rank = init_distributed()
    
    # Load Ming-omni model
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        load_image_gen=False,  # Don't need image generation feature
        low_cpu_mem_usage=True,
        device_map="auto"
    ).eval()
    
    # print(f"Model device map: {model.hf_device_map}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    print("Model loaded, starting evaluation...")
    
    # Process result writing directly in evaluate_model
    eval_results = evaluate_model(
        dataloader, 
        model, 
        processor, 
        output_file, 
        max_duration,
        max_frames=max_frames
    )

    print(f"\nEvaluation completed, results saved to {output_file}")
    print(f"Final statistics:")
    print(f"  - Total processed: {eval_results['total_processed']}")
    print(f"  - Correct answers: {eval_results['correct_answers']}")
    print(f"  - Final accuracy: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
