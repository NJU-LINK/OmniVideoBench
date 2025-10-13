import torch
import json
import os
import string
import re
import warnings
import sys
import ujson

from tqdm import tqdm
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

# Add project root to path for relative imports
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from dataloader import VideoQADaloader
from utils.utils import (
    clean_text,
    save_results,
    filter_qa_pairs_by_duration,
    load_existing_results,
    validate_qa_pair,
    extract_model_answer,
    handle_processing_error,
    set_seed
)

# Set environment variables
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'



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


def extract_answer(text: str) -> str:
    """Extracts content from the <answer> tag or finds option letters."""
    # First try to find answer tags
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for single letter answers (A, B, C, D, etc.)
    # Split by common delimiters and check for valid options
    parts = re.split(r'[:\s\n]', text)
    potential_answers = [p.strip() for p in parts if p.strip() and len(p.strip()) == 1 and p.strip().isalpha()]
    
    if potential_answers:
        return potential_answers[-1]  # Return the last likely option
        
    return text.strip()


def build_conversation(video_path, question, options):
    """Build conversation format for Baichuan-Omni model."""
    options_text = "\n".join(options)
    
    # Build prompt in English
    prompt = (
        "You are a video understanding assistant. Please answer the following question based on the video content:\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Please provide your final answer within the <answer> </answer> tags. "
        "Please provide only the single option letter (e.g., A, B, C, D, etc.)."
    )
    
    # Build input string with video (use 'local' key for video, not 'path')
    video_token = ujson.dumps({'local': video_path}, ensure_ascii=False)
    input_string = f"<B_SYS>You are Baichuan AI Assistant, an AI assistant developed by Baichuan Intelligence.<C_Q><video_start_baichuan>{video_token}<video_end_baichuan>{prompt}<C_A>"
    
    return input_string


def load_model_and_processor(model_path: str):
    """Load and initialize the Baichuan-Omni model and processor."""
    print("Loading model and processor...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Set to inference mode
    model.training = False
    
    # Bind processor
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    model.bind_processor(tokenizer, training=False, relative_path=cache_dir)
    
    print("Model loading completed!")
    return model, tokenizer


def process_single_qa_pair(qa_pair, model, tokenizer):
    """Process a single QA pair and return the result."""
    single_result = create_result_template(qa_pair)
    video_path = qa_pair.get("video_path")
    question = qa_pair.get("question")
    options = qa_pair.get("options")
    correct_answer = qa_pair.get("answer")
    duration = qa_pair.get("duration")
    
    # Validate QA pair
    if not validate_qa_pair(qa_pair):
        print(f"Skipping item due to missing data: {qa_pair}")
        return single_result
    
    # Check video file existence
    if not os.path.exists(video_path):
        print(f"Video file not found, skipping: {video_path}")
        single_result['model_answer'] = 'Video file not found.'
        return single_result
    
    try:
        print(f"Processing {video_path} (duration: {duration}s)...")
        
        # Build conversation
        input_string = build_conversation(video_path, question, options)
        print(f"Input string length: {len(input_string)}")
        
        # Process input
        print("Processing multimedia input...")
        ret = model.processor([input_string])
        
        # Prepare input data
        inputs = {
            'input_ids': ret.input_ids.cuda(),
            'attention_mask': ret.attention_mask.cuda() if ret.attention_mask is not None else None,
            'labels': ret.labels.cuda() if ret.labels is not None else None,
            'audios': ret.audios.cuda() if ret.audios is not None else None,
            'images': [torch.tensor(img, dtype=torch.float32).cuda() for img in ret.images] if ret.images is not None else None,
            'patch_nums': ret.patch_nums if ret.patch_nums is not None else None,
            'images_grid': ret.images_grid if ret.images_grid is not None else None,
            'videos': [torch.tensor(img, dtype=torch.float32).cuda() for img in ret.videos] if ret.videos is not None else None,
            'videos_patch_nums': ret.videos_patch_nums if ret.videos_patch_nums is not None else None,
            'videos_grid': ret.videos_grid if ret.videos_grid is not None else None,
            'encoder_length': ret.encoder_length.cuda() if ret.encoder_length is not None else None,
            'bridge_length': ret.bridge_length.cuda() if ret.bridge_length is not None else None,
        }
        
        # Remove None values
        inputs = {k: v for k, v in inputs.items() if v is not None}
        
        print("Starting inference...")
        input_length = ret.input_ids.shape[-1] if hasattr(ret, 'input_ids') and ret.input_ids is not None else 0
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_k=20,
                top_p=0.9,
                repetition_penalty=1.05,
                tokenizer=tokenizer,
                stop_strings=["<C_Q>", ".", ","],
                return_dict_in_generate=True,
            )
        
        print("Decoding response...")
        # Only decode newly generated tokens
        if hasattr(outputs, 'sequences'):
            generated_tokens = outputs.sequences[:, input_length:] if outputs.sequences.shape[-1] > input_length else outputs.sequences
        else:
            generated_tokens = outputs[:, input_length:] if outputs.shape[-1] > input_length else outputs
            
        response_text = tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"Model response: {response_text}")
        
        # Extract answer
        model_answer = extract_answer(response_text)
        single_result['model_answer'] = model_answer
        single_result['is_correct'] = clean_text(model_answer) == clean_text( correct_answer )
        
        print(f"Extracted answer: {model_answer}, Correct answer: {correct_answer}, Is correct: {single_result['is_correct']}")
        
    except Exception as e:
        print(f"Processing error: {e}")
        single_result['model_answer'] = handle_processing_error(e, video_path)
    
    return single_result


def calculate_and_print_progress(results, total_items, max_duration):
    """Calculate and print current progress."""
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


def run_baichuan_evaluation(model_path: str, data_json_file: str=None, video_dir: str=None, output_file: str=None, max_duration: int=None):
    """Run the evaluation on the dataset using the Baichuan-Omni model."""
    set_seed(42)

    # Initialize model and processor
    model, tokenizer = load_model_and_processor(model_path)
    print(model.hf_device_map)
    
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
    
    # Initialize results list
    # results = []
    results, processed_ids = load_existing_results(output_file)
    
    print(f"Starting evaluation on {len(filtered_qa_pairs)} QA pairs with {model_path}...")
    
    start_index = len(results)
    # Process each QA pair
    for qa_pair in tqdm(filtered_qa_pairs[start_index:], desc="Evaluating Baichuan-Omni"):
        single_result = process_single_qa_pair(qa_pair, model, tokenizer)
        
        if single_result is not None:
            results.append(single_result)
            
            # Save results after each processing
            save_results(results, output_file)
            
            # Print current progress
            calculate_and_print_progress(results, len(filtered_qa_pairs), max_duration)
    
    # Print final results
    print_final_results(results, max_duration, output_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Baichuan-Omni evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the Baichuan-Omni model")
    parser.add_argument("--data_json_file", type=str, required=True,
                       help="Path to the QA data JSON file")
    parser.add_argument("--video_dir", type=str, required=True,
                       help="Path to the video directory")
    parser.add_argument("--output_file", type=str,
                       help="Path to output results file (optional)")
    parser.add_argument("--max_duration", type=int, default=6000,
                       help="Maximum video duration in seconds (default: 6000)")
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output_file:
        model_name = os.path.basename(args.model_path)
        output_dir = os.path.join(project_root, "eval_results")
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, f"{model_name}_results.json")
    
    print(f"Model path: {args.model_path}")
    print(f"Data file: {args.data_json_file}")
    print(f"Video dir: {args.video_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Max duration: {args.max_duration}s")
    
    run_baichuan_evaluation(args.model_path, args.data_json_file, args.video_dir, args.output_file, args.max_duration)
