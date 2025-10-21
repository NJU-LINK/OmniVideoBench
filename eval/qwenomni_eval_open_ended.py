import torch
import json
import os
import string
import re
import sys

from tqdm import tqdm
import soundfile as sf
if not hasattr(sf, "SoundFileRuntimeError"):
    sf.SoundFileRuntimeError = RuntimeError

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
#from qwen_omni_utils import process_mm_info


import sys
# Add project root to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from utils.utils import (
    clean_text,
    create_unique_id,
    load_existing_results,
    save_results,
    filter_qa_pairs_by_duration,
    validate_qa_pair,
    extract_model_answer,
    handle_processing_error,
    set_seed,
    )

from utils.vision_process import process_vision_info
from utils.audio_process import process_audio_info

from dataloader import VideoQADaloader


os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def process_mm_info(conversations, use_audio_in_video, return_video_kwargs=False):
    audios = process_audio_info(conversations, use_audio_in_video)
    vision = process_vision_info(conversations, return_video_kwargs=return_video_kwargs)
    return (audios,) + vision
    

def load_model_and_processor(model_name: str):
    """Load and initialize the Qwen model and processor."""
    print("Loading model and processor...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # device_map = _device_map,
        attn_implementation="flash_attention_2",
        # load_in_8bit=True
    )
    print(model.hf_device_map)


    # processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    # model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda",attn_implementation="flash_attention_2",)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    return model, processor

def create_result_template(qa_pair):
    """Create a template result dictionary."""
    video_path = qa_pair.get("video_path")
    return {
        'video': os.path.basename(video_path) if video_path else 'N/A',
        'duration': qa_pair.get("duration"),
        'question': qa_pair.get("question"),
        'options': qa_pair.get("options"),
        'correct_answer': qa_pair.get("answer") or qa_pair.get("confused_options"),
        'model_answer': None,
        'is_correct': False,
    }


def build_conversation(video_path, question, options):
    """Build conversation format for the model."""
    options_text = "\n".join(options)
    prompt = (
        "### Instruction"
        "You are given a question about the following video."
        "Your task is to watch the video and answer ONLY with the final choice. "

        "## Requirements"
        "- Do NOT output reasoning, explanations, or analysis."
        "- Do NOT repeat the question."
        "- The answer must be the **shortest possible form**."
        "- Wrap your final answer inside LaTeX-style bbox for clarity."

        "### Question"
        f"{question}\n\n"

        "### Answer format"
        "\\bbox{{Final Answer}})"
    )

    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    return conversation, prompt

def get_list_memory_usage(lst):
    total_size = sys.getsizeof(lst)
    
    for item in lst:
        total_size += sys.getsizeof(item)
    
    return total_size

def process_multimedia_input(conversation, processor):
    """Process multimedia information from conversation."""
    USE_AUDIO_IN_VIDEO = True
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    print("processing video...")
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    # TODO: keep frames under 120
    MAX_FRAMES=120
    total_frames = len(videos)
    if total_frames > MAX_FRAMES:
        print(f"Video has {total_frames} frames, exceeding {MAX_FRAMES}. Sampling down...")
        # uniform sampling
        step = total_frames / MAX_FRAMES
        sampled_indices = [int(i * step) for i in range(MAX_FRAMES)]
        videos = [videos[i] for i in sampled_indices]
    
    print("frames are ready, start processing...")
    inputs = processor(
        text=text, 
        audio=audios,
        images=images, 
        videos=videos,
        return_tensors="pt", 
        padding=True, 
        use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    
    return inputs, USE_AUDIO_IN_VIDEO


def generate_model_response(model, inputs, use_audio_in_video):
    """Generate response from the model."""
    inputs = inputs.to(model.device).to(model.dtype)
    
    print("start inferencing...")
    with torch.no_grad():
        text_ids, audio = model.generate(
            **inputs, 
            use_audio_in_video=use_audio_in_video,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True
        )
    
    del inputs
    return text_ids, audio



def calculate_and_print_progress(valid_results, total_items, max_duration):
    """Calculate and print current progress."""
    current_correct = sum(1 for r in valid_results if r.get('is_correct', False))
    current_accuracy = current_correct / len(valid_results) if valid_results else 0


def print_final_results(valid_results, max_duration, output_file):
    """Print final evaluation results."""
    total_predictions = len(valid_results)
    if total_predictions > 0:
        final_correct = sum(1 for r in valid_results if r.get('is_correct', False))
        accuracy = final_correct / total_predictions
        print(f"\nEvaluation completed!")
        print(f"Total questions processed (duration < {max_duration}s): {total_predictions}")
        print(f"Correct predictions: {final_correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Detailed results saved to: {output_file}")
    else:
        print("No questions were processed.")

def extract_model_answer(response_text,prompt=None):
    """Extract and clean model answer from generated text."""
    # Extract the model's answer from the response
    if "assistant" in response_text:
        model_answer = response_text.split("assistant")[-1].strip()
    else:
        # If no "assistant" marker, take the last part after the prompt
        model_answer = response_text.split(prompt)[-1].strip()

    return model_answer

def process_single_qa_pair(qa_pair, model, processor, processed_ids):
    """Process a single QA pair and return the result."""
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
        print(f"Skipping item due to missing data: {qa_pair}")
        return single_result, False
    
    # Check video file existence
    if not os.path.exists(video_path):
        print(f"Video file not found, skipping: {video_path}")
        single_result['model_answer'] = 'Video file not found.'
        return single_result, False
    
    try:
        print(f"Processing {video_path} (duration: {duration}s)...")
        
        # Build conversation and process multimedia
        conversation, prompt = build_conversation(video_path, question, options)
        print("build the conversation...")
        inputs, use_audio_in_video = process_multimedia_input(conversation, processor)
        print("="*50)
        print("Total input tokens:", inputs["input_ids"].shape[1])
        print("="*50)
        print("process the multimedia input...")
        
        # Generate model response
        text_ids, audio = generate_model_response(model, inputs, use_audio_in_video)
        
        # Extract and evaluate answer
        print("decode the response...")
        response_text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        model_answer = extract_model_answer(response_text,prompt)
        single_result['model_answer'] = model_answer
        single_result['is_correct'] = clean_text(model_answer) == clean_text(correct_answer)
        
    except Exception as e:
        single_result['model_answer'] = handle_processing_error(e, video_path)
    
    return single_result, False


def run_qwen_evaluation(data_json_file: str, video_dir: str, output_file: str, max_duration: int, model_path: str):
    """Run the evaluation on the dataset using the Qwen-Omni model."""
    set_seed(42)

    # Initialize model and processor
    model, processor = load_model_and_processor(model_path)
    
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
    
    print(f"Starting evaluation on {len(filtered_qa_pairs)} QA pairs with {model_path}...")
    print(f"Resuming from {len(results)} existing results")

    start_index = len(results)
    
    # Process each QA pair
    for qa_pair in tqdm(filtered_qa_pairs[start_index:], desc="Evaluating Qwen-Omni"):

        print("start processing a new qa_pair...")
        print("="*50)
        single_result, was_skipped = process_single_qa_pair(qa_pair, model, processor, processed_ids)
        
        #if was_skipped:
            #continue

        if single_result is not None:
            results.append(single_result)
            
            # Save results after each processing
            save_results(results, output_file)
            
        # Print current progress
        valid_results = [r for r in results if not r.get('model_answer', '').startswith(('Video file not found', 'Error:'))]
        calculate_and_print_progress(valid_results, len(filtered_qa_pairs), max_duration)
    
    # Filter out skipped items (video not found or file errors)
    valid_results = [r for r in results if not r.get('model_answer', '').startswith(('Video file not found', 'Error:'))]
    
    # Print final results
    print_final_results(valid_results, max_duration, output_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Qwen-Omni model on video QA dataset (open-ended)")
    parser.add_argument("--data_json_file", type=str, required=True, help="Path to the data JSON file")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--output_file", type=str, default="./eval_results/qwen_omni_open_ended_output.json", help="Output file path for evaluation results")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Qwen-Omni model")
    parser.add_argument("--max_duration", type=int, default=6000, help="Maximum video duration in seconds")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    run_qwen_evaluation(args.data_json_file, args.video_dir, args.output_file, args.max_duration, args.model_path)
