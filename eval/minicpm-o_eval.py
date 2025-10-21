import os
import json
import math
import tempfile
from typing import List, Dict, Any
import gc
import psutil
import time
import argparse
import logging

import numpy as np
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
import librosa
import soundfile as sf
import torch
from transformers import AutoModel, AutoTokenizer
import random

import sys
# Add project root to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from dataloader import VideoQADaloader


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def filter_qa_pairs_by_duration(qa_pairs, max_duration):
    """
    Filter QA pairs based on video duration.
    
    Args:
        qa_pairs: List of QA pairs
        max_duration: Maximum video duration in seconds
        
    Returns:
        Filtered list of QA pairs
    """
    if max_duration is None:
        return qa_pairs
        
    filtered_qa_pairs = []
    for qa_pair in qa_pairs:
        duration = qa_pair.get("duration")
        if duration is not None and duration < max_duration:
            filtered_qa_pairs.append(qa_pair)
        elif duration is None:
            logging.warning(f"No duration data for video {qa_pair.get('video_path', 'Unknown')}, skipping")
    
    logging.info(f"Original dataset: {len(qa_pairs)} QA pairs")
    logging.info(f"Filtered dataset < {max_duration}s: {len(filtered_qa_pairs)} QA pairs")
    return filtered_qa_pairs


def get_video_chunk_content(video_path, flatten=False):
    """
    Extract video frames and audio content from video file with proper resource management.
    
    Args:
        video_path: Path to video file
        flatten: Whether to flatten the output list
        
    Returns:
        List of video chunks containing frames and audio segments
    """
    video = None
    try:
        video = VideoFileClip(video_path)
        logging.info(f'Video duration: {video.duration}s')
        
        # Check if video has audio track
        if video.audio is None:
            logging.warning(f"Warning: Video {video_path} has no audio track")
            sr = 16000
            audio_np = np.zeros(int(video.duration * sr), dtype=np.float32)
        else:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                    video.audio.write_audiofile(temp_audio_file.name, codec="pcm_s16le", fps=16000, logger=None)
                    audio_np, sr = librosa.load(temp_audio_file.name, sr=16000, mono=True)
            except Exception as e:
                logging.error(f"Audio extraction failed: {e}, using silent audio")
                sr = 16000
                audio_np = np.zeros(int(video.duration * sr), dtype=np.float32)
        
        # Ensure audio length matches video duration
        expected_audio_length = int(video.duration * sr)
        if len(audio_np) != expected_audio_length:
            logging.info(f"Adjusting audio length: {len(audio_np)} -> {expected_audio_length}")
            if len(audio_np) < expected_audio_length:
                audio_np = np.pad(audio_np, (0, expected_audio_length - len(audio_np)), 'constant')
            else:
                audio_np = audio_np[:expected_audio_length]
        
        num_units = math.ceil(video.duration)
        contents = []
        
        for i in range(num_units):
            start_idx = sr * i
            end_idx = min(sr * (i + 1), len(audio_np))
            if start_idx >= end_idx:
                continue

            # Use mid-point of segment for frame
            frame_time = min(i + 0.5, video.duration - 0.001)
            frame = video.get_frame(frame_time)
            image = Image.fromarray(frame.astype(np.uint8))
            
            audio = audio_np[start_idx:end_idx]
            if len(audio) < sr:
                audio = np.pad(audio, (0, sr - len(audio)), 'constant')

            if flatten:
                contents.extend(["<unit>", image, audio.astype(np.float32)])
            else:
                contents.append(["<unit>", image, audio.astype(np.float32)])

        logging.info(f"Processed {len(contents)} valid segments")
        return contents
        
    finally:
        if video:
            video.close()


def write_result_to_file(result, output_file):
    """Write a single result to JSON file"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    existing_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_results = []
    
    existing_results.append(result)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)


def load_processed_results(output_file):
    """
    Load already processed results to enable resume functionality.
    
    Returns:
        Set of processed item identifiers
    """
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            for item in existing_results:
                # Create unique ID from video name and first option
                video_name = item.get('video', '')
                options = item.get('options', [])
                if video_name and options:
                    unique_id = f"{video_name}::{options[0]}"
                    processed_ids.add(unique_id)
                    
            logging.info(f"Loaded {len(processed_ids)} already processed items from {output_file}")
        except (json.JSONDecodeError, FileNotFoundError):
            logging.warning(f"Output file {output_file} exists but cannot be parsed, starting fresh")
            processed_ids = set()
    
    return processed_ids


def evaluate_model(dataloader, model, tokenizer, ref_audio_path, output_file, max_duration=None):
    """
    Evaluate MiniCPM-O model on video QA dataset with memory and resource optimization.
    
    Args:
        dataloader: VideoQADaloader instance
        model: MiniCPM-O model
        tokenizer: Text tokenizer
        ref_audio_path: Path to reference audio for system prompt
        output_file: Path to save results
        max_duration: Maximum video duration to process (in seconds)
    """
    # Load reference audio for system prompt
    ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
    sys_msg = model.get_sys_prompt(ref_audio=ref_audio, mode='omni')

    # Get and filter QA pairs
    qa_pairs = dataloader.get_all_qa_pairs()
    qa_pairs = filter_qa_pairs_by_duration(qa_pairs, max_duration)

    # Load processed items for resume
    processed_ids = load_processed_results(output_file)
    
    logging.info(f"Starting evaluation, initial memory usage: {psutil.virtual_memory().percent}%")
    logging.info(f"Total QA pairs: {len(qa_pairs)}")
    torch.cuda.empty_cache()
    gc.collect()

    processed_count = 0
    error_count = 0
    skipped_count = 0
    correct_count = 0

    for idx, item in enumerate(qa_pairs):
        loop_start_time = time.time()

        # Check if already processed
        video_path = item['video_path']
        video_name = os.path.basename(video_path)
        options = item.get('options', [])
        
        if options:
            unique_id = f"{video_name}::{options[0]}"
            if unique_id in processed_ids:
                skipped_count += 1
                continue
        
        logging.info(f"\nProcessing {idx+1}/{len(qa_pairs)}: {video_name}")
        
        # Initialize result object
        result = {
            "video": video_name,
            "duration": item.get("duration", 0),
            "question": item.get("question", ""),
            "options": options,
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
            # Process video content
            video_proc_start = time.time()
            contents = get_video_chunk_content(video_path)
            logging.info(f"Video preprocessing time: {time.time() - video_proc_start:.2f}s")
            
            if not contents:
                raise ValueError("Video content extraction failed")

            # Build prompt
            question = item.get("question")
            correct_answer = item.get("answer")
            options_text = "\n".join(options)

            question_prompt = (
                "You are given a video. Based on the content of the video, answer the following question:\n\n"
                f"Question:\n{question}\n\n"
                f"Options:\n{options_text}\n\n"
                "Answer with the option's letter directly (e.g., A, B, C, or D). "
                "If your access to the video content is limited, at least one option that is more likely than the others must be chosen. "
                "Do not give any other reason for not choosing!"
            )

            contents.append(question_prompt)
            msgs = [sys_msg, {"role": "user", "content": contents}]

            # Model inference
            model_infer_start = time.time()
            res = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.5,
                max_new_tokens=512,
                omni_input=True,
                use_tts_template=False,
                generate_audio=False,
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True
            )
            logging.info(f"Model inference time: {time.time() - model_infer_start:.2f}s")
            
            model_answer = res.get("text", str(res))
            is_correct = (model_answer.strip().upper() == correct_answer.strip().upper())
            
            result["model_answer"] = model_answer
            result["is_correct"] = is_correct
            
            if is_correct:
                correct_count += 1
            
            processed_count += 1
            
        except Exception as e:
            error_msg = f"Error: {type(e).__name__} - {str(e)}"
            logging.error(f"Error processing video {video_name}: {error_msg}")
            result["model_answer"] = error_msg
            result["is_correct"] = False
            error_count += 1
            
            # Clear memory if CUDA/memory error
            if 'cuda' in str(e).lower() or 'memory' in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
        
        # Write result
        io_start_time = time.time()
        write_result_to_file(result, output_file)
        logging.info(f"File write time: {time.time() - io_start_time:.2f}s")
        
        # Explicit cleanup to help garbage collection
        if 'contents' in locals():
            del contents
        if 'msgs' in locals():
            del msgs
        if 'res' in locals():
            del res
        
        logging.info(f"Result written: {video_name} - {result['model_answer'][:50]}...")
        
        # Periodic memory cleanup
        if (idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            logging.info(f"Memory cleanup after {idx+1} videos, current usage: {psutil.virtual_memory().percent}%")
        
        logging.info(f"Total time for this video: {time.time() - loop_start_time:.2f}s")
    
    # Final statistics
    logging.info(f"\nEvaluation complete!")
    logging.info(f"Successfully processed: {processed_count} videos")
    logging.info(f"Errors: {error_count}")
    logging.info(f"Skipped (already processed): {skipped_count}")
    if processed_count > 0:
        logging.info(f"Accuracy: {(correct_count / processed_count) * 100:.2f}% ({correct_count}/{processed_count})")


def main():
    parser = argparse.ArgumentParser(description='MiniCPM-O Evaluation Script')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to MiniCPM-O model')
    parser.add_argument('--input_file', type=str, default='data.json',
                       help='Path to input QA JSON file')
    parser.add_argument('--video_dir', type=str, default='./videos',
                       help='Directory containing video files')
    parser.add_argument('--ref_audio', type=str, required=True,
                       help='Path to reference audio file for system prompt')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save results (default: {model_name}_results.json)')
    parser.add_argument('--max_duration', type=int, default=None,
                       help='Maximum video duration to process (in seconds)')
    parser.add_argument('--log_file', type=str, default='minicpm_o_eval.log',
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
    logging.info(f"Reference audio: {args.ref_audio}")
    logging.info(f"Output file: {args.output_file}")
    logging.info(f"Max duration: {args.max_duration}s" if args.max_duration else "Max duration: No limit")
    
    # Load dataloader
    dataloader = VideoQADaloader(args.input_file, args.video_dir)
    
    # Load model
    logging.info("Loading model...")
    os.environ['TRANSFORMERS_CACHE'] = args.model_path
    
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=True,
        local_files_only=False
    ).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    logging.info("Model loaded successfully")
    
    # Run evaluation
    evaluate_model(dataloader, model, tokenizer, args.ref_audio, args.output_file, args.max_duration)
    
    logging.info(f"\nEvaluation complete! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()