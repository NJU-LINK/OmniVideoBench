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
from decord import VideoReader, cpu
import cv2
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import random

import sys
# Add project root to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from dataloader import VideoQADaloader

import torch.distributed as dist

# dist.init_process_group(backend="nccl")
# torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# Set environment variables
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Add problematic videos blacklist
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
    
    # If not found, return the first character of original response (if A-D)
    first_char = response[0].upper() if response else ""
    if first_char in ['A', 'B', 'C', 'D']:
        return first_char
    
    return ""

def load_video_opencv(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """Alternative video loading using OpenCV"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"OpenCV cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f'Total video frames: {total_frames}, FPS: {fps}')
        
        # Calculate frame indices
        frame_indices = get_index(bound, fps, total_frames - 1, first_idx=0, num_segments=num_segments)
        
        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        
        for frame_index in frame_indices:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Cannot read frame {frame_index}, skipping")
                continue
                
            # Convert color space BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Dynamic preprocessing
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        
        cap.release()
        
        if not pixel_values_list:
            raise Exception("Failed to read any video frames")
            
        pixel_values = torch.cat(pixel_values_list)
        
        print(f'Extracted {len(frame_indices)} frames, total image patches: {pixel_values.shape[0]}')
        return pixel_values, num_patches_list
        
    except Exception as e:
        print(f"OpenCV video loading failed: {e}")
        raise e

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """Load video and extract frames, supports DECORD and OpenCV alternatives"""
    try:
        # First try using DECORD
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        print(f'Total video frames: {max_frame + 1}, FPS: {fps}')

        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        
        print(f'Extracted {len(frame_indices)} frames, total image patches: {pixel_values.shape[0]}')
        return pixel_values, num_patches_list
        
    except Exception as decord_error:
        print(f"DECORD loading failed: {decord_error}")
        
        # Check if it's a specific DECORD error
        error_msg = str(decord_error)
        if ("cannot find video stream" in error_msg or
            "DECORDError" in error_msg or
            "Invalid NAL unit size" in error_msg or
            "Error splitting the input into NAL units" in error_msg or
            "Error sending packet" in error_msg or
            "avcodec_send_packet" in error_msg or
            "DECORD loading failed" in error_msg):
            
            print("Detected DECORD compatibility issue, trying OpenCV alternative...")
            try:
                return load_video_opencv(video_path, bound, input_size, max_num, num_segments)
            except Exception as opencv_error:
                print(f"OpenCV alternative also failed: {opencv_error}")
                # Throw original DECORD error to maintain error message consistency
                raise decord_error
        else:
            # For other types of errors, throw directly
            raise decord_error

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
    print(f"Filtered dataset < {max_duration} seconds: {len(filtered_qa_pairs)} QA pairs")
    return filtered_qa_pairs

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

def evaluate_model(dataloader, model, tokenizer, output_file, max_duration, num_segments=8, max_num=1):
    """Evaluate using InternVL model"""
    # Use filter_qa_pairs_by_duration to filter QA pairs
    qa_pairs = filter_qa_pairs_by_duration(dataloader.get_all_qa_pairs(), max_duration)
    
    # Memory monitoring
    print(f"Starting evaluation, initial memory usage: {psutil.virtual_memory().percent}%")
    print(f"Filtered QA pairs count: {len(qa_pairs)}")
    torch.cuda.empty_cache()
    gc.collect()

    processed_count = 0
    error_count = 0
    correct_count = 0
    generation_config = dict(max_new_tokens=1024, do_sample=True)

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
            print(f"Video does not exist: {video_path}")
            result["model_answer"] = f"Error: Video file does not exist - {video_path}"
            write_result_to_file(result, output_file)
            error_count += 1
            continue

        try:
            # Load video and extract frames
            pixel_values, num_patches_list = load_video(
                video_path,
                num_segments=num_segments,
                max_num=max_num
            )
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            
            # Validate content integrity
            if pixel_values.size(0) == 0:
                print(f"Video content extraction failed, skipping: {video_name}")
                result["model_answer"] = "Error: Video content extraction failed"
                write_result_to_file(result, output_file)
                PROBLEMATIC_VIDEOS.add(video_path)
                error_count += 1
                continue
            
            print(f"Video processing completed: {pixel_values.shape[0]} image patches")

            question = item.get("question")
            correct_answer = item.get("answer")  # Fix: use "answer" instead of "correct_option"
            #TODO: Remove
            print(f"question: {question}")
            print(f"correct_answer: {correct_answer}")

            # Build video prefix
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            
            question_prompt = (
                f"{video_prefix}"
                "You are given a video. Based on the content of the video, answer the following question:\n\n"
                f"Question:\n{question}\n\n"
                f"Options:\n{options_text}\n\n"
                "Answer with the option's letter directly(e.g., A, B, C, or D). "
                "If your access to the video content is limited, at least one option that is more likely than the others must be chosen. "
                "Mustn't give any other reason for can not choose!"
            )

            # Model inference with error handling
            try:
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    question_prompt,
                    generation_config,
                    num_patches_list=num_patches_list
                )
                
                print(f"response: {response}")

                # Use regex to extract answer letter
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
        
        # Regular memory cleanup
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
    
    # Calculate accuracy
    accuracy = correct_count / processed_count if processed_count > 0 else 0.0
    
    print(f"\nEvaluation completed!")
    print(f"Successfully processed: {processed_count} videos")
    print(f"Correct answers: {correct_count}")
    print(f"Error count: {error_count}")
    print(f"Problematic videos count: {len(PROBLEMATIC_VIDEOS)}")
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
    parser.add_argument("--video_dir", type=str, required=True, help="Path to the video directory")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--max_duration", type=int, default=6000, help="Maximum video duration in seconds")
    parser.add_argument("--num_segments", type=int, default=128, help="Number of video segments")
    parser.add_argument("--max_num", type=int, default=1, help="Maximum number of frames")
    args = parser.parse_args()

    data_json_file = args.data_json_file
    video_dir = args.video_dir
    output_file = args.output_file
    model_path = args.model_path
    max_duration = args.max_duration
    num_segments = args.num_segments
    max_num = args.max_num
    
    dataloader = VideoQADaloader(data_json_file, video_dir)
    
    # Load InternVL model
    print("Loading InternVL model...")
    def init_distributed():
        if not dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Automatically set to 0/1 by accelerate (2 GPUs)
            torch.cuda.set_device(local_rank)  # Bind current process to corresponding GPU
            # Initialize process group (backend uses nccl, suitable for GPU communication)
            dist.init_process_group(backend="nccl")
        return int(os.environ.get("LOCAL_RANK", 0))

    # 2. First initialize distributed, then load model
    local_rank = init_distributed()
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
        #device_map={"": local_rank}
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    print("Model loaded, starting evaluation...")
    
    # Directly handle result writing in evaluate_model
    eval_results = evaluate_model(
        dataloader,
        model,
        tokenizer,
        output_file,
        max_duration,
        num_segments=num_segments,
        max_num=max_num
    )

    print(f"\nEvaluation completed, results saved to {output_file}")
    print(f"Final statistics:")
    print(f"  - Total processed: {eval_results['total_processed']}")
    print(f"  - Correct answers: {eval_results['correct_answers']}")
    print(f"  - Final accuracy: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
