import torch
import json
import os
import string
import re
import signal
import time
import gc
import psutil
from multiprocessing import Process, Queue
from contextlib import contextmanager

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
#from qwen_vl_utils import process_vision_info
import sys

# sys.path.append("../../")
sys.path.append("/root/siton-tmp/code/omni-bench")
from dataloader import VideoQADaloader


# sys.path.append("../../utils/")
sys.path.append("/root/siton-tmp/code/omni-bench")

from utils.vl_vision_process import process_vision_info

from utils.utils import (
    clean_text,
    create_unique_id,
    load_existing_results,
    save_results,
    filter_qa_pairs_by_duration,
    validate_qa_pair,
    extract_model_answer,
    handle_processing_error,
    set_seed
)
# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

MAX_CONTEXT_LENGTH=32768  # Qwen2.5-VL 最大上下文长度

@contextmanager
def timeout_context(timeout_seconds):
    """Context manager for implementing timeout mechanism."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
    
    # Set up the timeout signal
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        yield
    finally:
        # Reset the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def check_memory_usage():
    """Check current memory usage and return percentage."""
    memory = psutil.virtual_memory()
    return memory.percent

def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def is_video_corrupted(video_path):
    """Simple check to see if video file might be corrupted."""
    try:
        if not os.path.exists(video_path):
            return True, "File does not exist"
        
        file_size = os.path.getsize(video_path)
        if file_size < 1024:  # Less than 1KB is suspicious
            return True, "File too small"
        
        return False, "File appears normal"
    except Exception as e:
        return True, f"Error checking file: {str(e)}"

def load_qwenvl_model_and_processor(model_name):
    """Load Qwen2.5-VL model and processor."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
        
    )
    
    processor = AutoProcessor.from_pretrained(model_name,max_pixels = 640*28*28)
    
    return model, processor


def create_result_template(qa_pair):
    """Create a template result dictionary."""
    video_path = qa_pair.get("video_path")
    return {
        'video': os.path.basename(video_path) if video_path else 'N/A',
        'duration': qa_pair.get("duration"),
        'question': qa_pair.get("question"),
        'options': qa_pair.get("options"),
        'correct_answer': qa_pair.get("answer"),
        'model_answer': None,
        'is_correct': False,
    }


def build_qwenvl_conversation(video_path, question, options, asr_text=None):
    """
    Build conversation format for Qwen2.5-VL model.
    Optionally includes ASR text if provided.
    """
    options_text = "\n".join(options)

    # ASR text block, inserted if asr_text is provided
    asr_block = ""
    if asr_text and asr_text.strip():
        asr_block = f"[ASR Transcription]\n{asr_text}\n\n"

    # Construct the prompt with the optional ASR block
    prompt = (
        "You are given a video. Based on the content of the video, answer the following question:\n\n"
        f"The ASR transcription results for this video's audio are as follows: \n{asr_block}"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Answer with the option's letter directly(e.g., A, B, C, or D)."
        "If your access to the video content is limited, at least one option that is more likely than the others must be chosen."
        "Mustn't give any other reason for can not choose!"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages, prompt



def process_qwenvl_input(messages, processor):
    """Process input for Qwen2.5-VL model."""
    print("processing video...")
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    # TODO: 这里按照单卡上限对长视频进行采样，保持帧数不超过120
    MAX_FRAMES=120
    total_frames = len(video_inputs[0])
    print("total_frames:", total_frames)
    # print("type of video_inputs:", video_inputs)
    # print("len of video_inputs[0]:", len(video_inputs[0]))
    if total_frames > MAX_FRAMES:
        # 均匀采样
        print("超过最大帧限制，开始均匀采帧")
        step = total_frames / MAX_FRAMES
        sampled_indices = [int(i * step) for i in range(MAX_FRAMES)]
        videos = [video_inputs[0][i] for i in sampled_indices]


    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    return inputs


def generate_qwenvl_response(model, inputs):
    """Generate response from Qwen2.5-VL model."""
    inputs = inputs.to(model.device)
    
    print("start inferencing...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    return generated_ids


def calculate_and_print_progress(results, total_items, max_duration):
    """Calculate and print current progress."""
    # print("type of results", type(results))
    # print("type of results[0]", type(results))
    current_correct = sum(1 for r in results if r.get('is_correct', False))
    current_accuracy = current_correct / len(results) if results else 0
    print(f"Progress: {len(results)}/{total_items}, Current accuracy: {current_accuracy:.2%} ({current_correct}/{len(results)})")


def print_final_results(results, max_duration, output_file):
    """Print final evaluation results."""
    total_predictions = len(results)
    if total_predictions > 0:
        final_correct = sum(1 for r in results if r.get('is_correct', False))
        accuracy = final_correct / total_predictions
        print(f"\nEvaluation completed!")
        print(f"Total questions processed (duration < {max_duration}s): {total_predictions}")
        print(f"Correct predictions: {final_correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Detailed results saved to: {output_file}")
    else:
        print("No questions were processed.")


def process_single_qa_pair(qa_pair, model, processor, processed_ids):
    """Process a single QA pair and return the result with enhanced error handling."""
    unique_id = create_unique_id(qa_pair)
    
    # Skip if already processed
    if unique_id in processed_ids:
        print(f"Skipping already processed item: {unique_id}")
        return None, True  # None result, but processed
    
    single_result = create_result_template(qa_pair)
    video_path = qa_pair.get("video_path")
    question = qa_pair.get("question")
    options = qa_pair.get("options")
    correct_answer = qa_pair.get("answer")
    duration = qa_pair.get("duration")
    
    # Validate QA pair
    if not validate_qa_pair(qa_pair):
        print(f"❌ Skipping item due to missing data: {qa_pair}")
        single_result['model_answer'] = 'Missing required data fields'
        single_result['error_type'] = 'validation_error'
        return single_result, False
    
    # Enhanced video file checks
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        single_result['model_answer'] = 'Video file not found.'
        single_result['error_type'] = 'file_not_found'
        return single_result, False
    
    # Check if video might be corrupted
    is_corrupted, corruption_reason = is_video_corrupted(video_path)
    if is_corrupted:
        print(f"❌ Video file appears corrupted: {video_path}, reason: {corruption_reason}")
        single_result['model_answer'] = f'Video file corrupted: {corruption_reason}'
        single_result['error_type'] = 'file_corrupted'
        return single_result, False
    
    # Check memory usage before processing
    memory_percent = check_memory_usage()
    if memory_percent > 90:
        print(f"⚠️ High memory usage ({memory_percent:.1f}%), clearing cache...")
        clear_gpu_cache()
    
    try:
        print(f"🔄 Processing {video_path} (duration: {duration}s)...")
        
        # Use timeout mechanism for video processing
        with timeout_context(300):  # 5 minutes timeout
            # Build conversation and process input
            # === START: DYNAMIC ASR INTEGRATION LOGIC ===

            # 1. First, process the input WITHOUT ASR to get the baseline token count
            initial_messages, _ = build_qwenvl_conversation(video_path, question, options, asr_text=None)
            initial_inputs = process_qwenvl_input(initial_messages, processor)
            baseline_token_count = initial_inputs["input_ids"].shape[1]
            
            print(f"Baseline token count (video + base prompt): {baseline_token_count}")

            # 2. Calculate available space for ASR text
            available_space = MAX_CONTEXT_LENGTH - baseline_token_count
            final_asr_text = None
            
            if available_space > 100: # Set a threshold to make it worthwhile
                # 3. Try to load ASR text
                asr_text_to_add = ""
                try:
                    video_base = os.path.splitext(os.path.basename(video_path))[0]
                    asr_path = f"/root/siton-tmp/code/asr/asr_results/{video_base}.txt"
                    if os.path.exists(asr_path):
                        with open(asr_path, "r", encoding="utf-8") as f:
                            asr_text_to_add = f.read().strip()
                except Exception as e:
                    print(f"⚠️ Failed to read ASR file for {video_path}: {e}")

                if asr_text_to_add:
                    # 4. Tokenize and truncate ASR text to fit the available space
                    asr_tokens = processor.tokenizer.encode(asr_text_to_add, add_special_tokens=False)
                    
                    if len(asr_tokens) > available_space:
                        print(f"ASR text is too long ({len(asr_tokens)} tokens). Truncating to fit {available_space} tokens.")
                        truncated_tokens = asr_tokens[:available_space]
                    else:
                        truncated_tokens = asr_tokens

                    # 5. Decode truncated tokens back to string for clean insertion
                    final_asr_text = processor.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    print(f"Successfully loaded and truncated ASR. Adding ~{len(truncated_tokens)} tokens.")
            else:
                print(f"Not enough space for ASR text. Available: {available_space} tokens.")


            # 6. Build final prompt and process inputs (with or without ASR)
            final_messages, prompt = build_qwenvl_conversation(video_path, question, options, asr_text=final_asr_text)
            inputs = process_qwenvl_input(final_messages, processor)
            
            # === END: DYNAMIC ASR INTEGRATION LOGIC ===
            seq_len = inputs["input_ids"].shape[1]
            print("="*50)
            print("每个样本的总 token 数:", seq_len)
            print("="*50)
            
            # Generate model response with timeout
            generated_ids = generate_qwenvl_response(model, inputs)
            
            # Extract and evaluate answer
            print("🔄 decode the response...")
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            model_answer = extract_model_answer(response_text, prompt)
            single_result['model_answer'] = model_answer
            single_result['is_correct'] = clean_text(model_answer) == clean_text(correct_answer)
            
        print(f"✅ Successfully processed {os.path.basename(video_path)}")
        
    except TimeoutError as e:
        print(f"⏰ Timeout processing {video_path}: {e}")
        single_result['model_answer'] = f'Processing timeout after 5 minutes'
        single_result['error_type'] = 'timeout_error'
        single_result['error_details'] = str(e)
        clear_gpu_cache()  # Clear cache after timeout
        
    except Exception as e:
        # Enhanced error handling with categorization
        error_str = str(e).lower()
        error_type = type(e).__name__
        
        print(f"❌ Error processing {video_path}: {error_type}: {e}")
        
        # Categorize different types of errors
        if any(keyword in error_str for keyword in ['nal unit', 'invalid data', 'decord', 'avcodec', 'video', 'ffmpeg']):
            if 'nal unit' in error_str or 'invalid data' in error_str:
                single_result['model_answer'] = 'Video corrupted: Unable to decode video frames'
                single_result['error_type'] = 'video_decode_failure'
            elif 'decord' in error_str or 'ffmpeg' in error_str:
                single_result['model_answer'] = 'Video backend error: Decoder failure'
                single_result['error_type'] = 'decoder_backend_failure'
            else:
                single_result['model_answer'] = f'Video processing error: {error_type}'
                single_result['error_type'] = 'video_processing_error'
        elif 'out of memory' in error_str or 'cuda' in error_str:
            single_result['model_answer'] = 'GPU memory insufficient for video processing'
            single_result['error_type'] = 'memory_error'
            clear_gpu_cache()  # Clear cache after memory error
        else:
            single_result['model_answer'] = handle_processing_error(e, video_path)
            single_result['error_type'] = 'general_error'
        
        single_result['error_details'] = str(e)[:200]  # Truncate long error messages
    
    return single_result, False


def run_qwenvl_evaluation(data_json_file: str, video_dir: str, model_name: str, output_file: str, max_duration: int):
    """Run the evaluation on the dataset using the Qwen2.5-VL model."""
    set_seed(42)

    # Initialize model and processor
    model, processor = load_qwenvl_model_and_processor(model_name)
    
    # Load and filter data
    dataloader = VideoQADaloader(data_json_file=data_json_file, video_dir=video_dir)
    all_qa_pairs = dataloader.get_all_qa_pairs()
    if not all_qa_pairs:
        print("Failed to load data or no QA pairs found. Exiting evaluation.")
        return
    
    filtered_qa_pairs = filter_qa_pairs_by_duration(all_qa_pairs, max_duration)
    if not filtered_qa_pairs:
        print(f"No videos with duration < {max_duration} seconds found. Exiting evaluation.")
        return
    
    # Load existing results for resume functionality
    results, processed_ids = load_existing_results(output_file)
    
    print(f"Starting evaluation on {len(filtered_qa_pairs)} QA pairs with {model_name}...")
    print(f"Resuming from {len(results)} existing results")
    
    # Process each QA pair with enhanced monitoring
    processed_count = 0
    error_count = 0
    
    start_index = len(results)
    for qa_pair in tqdm(filtered_qa_pairs[start_index:], desc="Evaluating Qwen2.5-VL"):
        try:
            single_result, was_skipped = process_single_qa_pair(qa_pair, model, processor, processed_ids)
            
            if single_result is not None:
                results.append(single_result)
                processed_count += 1
                
                # Track errors
                if single_result.get('error_type'):
                    error_count += 1
                    print(f"⚠️ Error count: {error_count}/{processed_count}")
                
                # Save results after each processing (critical for resume capability)
                save_results(results, output_file)
                
                # Print current progress
                calculate_and_print_progress(results, len(filtered_qa_pairs), max_duration)
                
                # Periodic memory and cache cleanup (every 10 videos)
                if processed_count % 10 == 0:
                    memory_percent = check_memory_usage()
                    print(f"💾 Memory usage: {memory_percent:.1f}%")
                    if memory_percent > 85:
                        print("🧹 Performing cleanup...")
                        clear_gpu_cache()
                        
        except KeyboardInterrupt:
            print("\n⚠️ Evaluation interrupted by user. Saving current results...")
            save_results(results, output_file)
            print(f"💾 Results saved to {output_file}")
            break
            
        except Exception as e:
            print(f"❌ Critical error in main loop: {type(e).__name__}: {e}")
            # Save current results before potentially exiting
            save_results(results, output_file)
            
            # Don't exit immediately, try to continue with next video
            error_count += 1
            if error_count > len(filtered_qa_pairs) * 0.5:  # If more than 50% errors
                print("❌ Too many errors encountered. Stopping evaluation.")
                break
    
    # Print final results
    print_final_results(results, max_duration, output_file)

# def run_qwenvl_evaluation(data_json_file: str, video_dir: str, output_file: str, max_duration: int):
#     """Run the evaluation on the dataset using the Qwen2.5-VL model."""
#     set_seed(42)

#     # Initialize model and processor
#     MODEL_NAME = "/fs-computility/llm_code_collab/liujiaheng/caoruili/models/Qwen2.5-VL-7B-Instruct"
#     model, processor = load_qwenvl_model_and_processor(MODEL_NAME)
    
#     # Load and filter data
#     dataloader = VideoQADaloader(data_json_file=data_json_file, video_dir=video_dir)
#     all_qa_pairs = dataloader.get_all_qa_pairs()
#     if not all_qa_pairs:
#         print("Failed to load data or no QA pairs found. Exiting evaluation.")
#         return
    
#     filtered_qa_pairs = filter_qa_pairs_by_duration(all_qa_pairs, max_duration)
    
#     if not filtered_qa_pairs:
#         print(f"No videos with duration < {max_duration} seconds found. Exiting evaluation.")
#         return
    
#     # Load existing results for resume functionality
#     results, processed_ids = load_existing_results(output_file)
    
#     # --- REFINED LOGIC FOR RESUMING ---
#     # Filter out the QA pairs that have already been processed
#     qa_pairs_to_process = [
#         qa for qa in filtered_qa_pairs if create_unique_id(qa) not in processed_ids
#     ]
    
#     total_qa_count = len(filtered_qa_pairs)
#     remaining_count = len(qa_pairs_to_process)
    
#     print(f"Found {total_qa_count} total QA pairs.")
#     print(f"Resuming from {len(results)} existing results. {remaining_count} items remaining to process.")
    
#     if not qa_pairs_to_process:
#         print("No new items to process. Evaluation may be complete.")
#     else:
#         # Process each remaining QA pair
#         for qa_pair in tqdm(qa_pairs_to_process, desc="Evaluating Qwen2.5-VL"):
#             single_result = process_single_qa_pair(qa_pair, model, processor, processed_ids)
            
#             if single_result is not None:
#                 results.append(single_result)
                
#                 # Save results after each processing
#                 save_results(results, output_file)
                
#                 # Print current progress
#                 calculate_and_print_progress(results, total_qa_count, max_duration)
    
#     # Print final results
#     print_final_results(results, max_duration, output_file)

if __name__ == "__main__":
    task_name = "6000_second_batch"
    # base_dir = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench"
    # data_json_file=f"{base_dir}/data/merged_qas_1_0817.json"
    data_json_file = "/root/siton-tmp/code/omni-bench/final_data/qa_data.json"
    video_dir="/root/siton-tmp/omni-videos/omni_videos_v3"
    
    model_name = "/root/siton-tmp/code/models/Qwen_Qwen2.5-VL-72B-Instruct"
    model_basename = os.path.basename(model_name)
    output_file = f"/root/siton-tmp/code/omni-bench/eval_results/qwen_vl_72B_ASR_qa_data.json"  

    max_duration = 6000
    
    run_qwenvl_evaluation(data_json_file, video_dir, model_name, output_file, max_duration)