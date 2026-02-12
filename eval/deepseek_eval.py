import os
import json
import argparse
import sys
from openai import OpenAI
from tqdm import tqdm

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
    set_seed
)
from dataloader import VideoQADaloader


def get_api_config():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")
    
    if not api_key:
        raise ValueError(
            "Please set DEEPSEEK_API_KEY environment variable before running.\n"
            "Example: export DEEPSEEK_API_KEY='sk-xxx'"
        )
    
    return api_key, base_url, model_name


def init_client():
    api_key, base_url, model_name = get_api_config()
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_name


def create_result_template(qa_pair):
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


def build_prompt(question, options):
    options_text = "\n".join(options)
    prompt = (
        "You are given a video. Based on the content of the video, answer the following question:\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Answer with the option's letter directly (e.g., A, B, C, or D)."
        "If your access to the video content is limited, at least one option that is more likely than the others must be chosen."
        "Mustn't give any other reason for can not choose!"
    )
    return prompt


def calculate_and_print_progress(results, total_items):
    current_correct = sum(1 for r in results if r.get('is_correct', False))
    current_accuracy = current_correct / len(results) if results else 0
    print(f"Progress: {len(results)}/{total_items}, Accuracy: {current_accuracy:.2%} ({current_correct}/{len(results)})")


def print_final_results(results, max_duration, output_file):
    total_predictions = len(results)
    if total_predictions > 0:
        final_correct = sum(1 for r in results if r.get('is_correct', False))
        accuracy = final_correct / total_predictions
        print(f"\n=== Final Results ===")
        print(f"Total questions processed (duration < {max_duration}s): {total_predictions}")
        print(f"Correct predictions: {final_correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Detailed results saved to: {output_file}")
    else:
        print("No questions were processed.")


def process_single_qa_pair(qa_pair, client, model_name, processed_ids):
    unique_id = create_unique_id(qa_pair)
    
    if unique_id in processed_ids:
        print(f"Skipping already processed item: {unique_id}")
        return None, True
    
    single_result = create_result_template(qa_pair)
    question = qa_pair.get("question")
    options = qa_pair.get("options")
    correct_answer = qa_pair.get("answer")
    
    if not validate_qa_pair(qa_pair):
        print(f"Skipping item due to missing data")
        single_result['model_answer'] = 'Missing required data fields'
        return single_result, False
    
    try:
        prompt = build_prompt(question, options)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        model_answer = response.choices[0].message.content.strip()
        model_answer = extract_model_answer(model_answer, prompt)
        
        single_result['model_answer'] = model_answer
        single_result['is_correct'] = clean_text(model_answer) == clean_text(correct_answer)
        
        if single_result['is_correct']:
            print(f"Correct for video: {single_result['video']}")
        else:
            print(f"Incorrect for video: {single_result['video']}")
            print(f"  Expected: {correct_answer}, Got: {model_answer}")
            
    except Exception as e:
        print(f"Error processing: {e}")
        single_result['model_answer'] = f"Error: {type(e).__name__} - {str(e)}"
    
    return single_result, False


def run_deepseek_evaluation(data_json_file: str, video_dir: str, output_file: str, max_duration: int):
    set_seed(42)
    
    client, model_name = init_client()
    print(f"Using model: {model_name}")
    
    dataloader = VideoQADaloader(data_json_file=data_json_file, video_dir=video_dir)
    all_qa_pairs = dataloader.get_all_qa_pairs()
    
    if not all_qa_pairs:
        print("Failed to load data or no QA pairs found. Exiting.")
        return
    
    filtered_qa_pairs = filter_qa_pairs_by_duration(all_qa_pairs, max_duration)
    if not filtered_qa_pairs:
        print(f"No videos with duration < {max_duration}s found. Exiting.")
        return
    
    results, processed_ids = load_existing_results(output_file)
    
    print(f"Starting evaluation on {len(filtered_qa_pairs)} QA pairs...")
    print(f"Resuming from {len(results)} existing results")
    
    start_index = len(results)
    
    for qa_pair in tqdm(filtered_qa_pairs[start_index:], desc="Evaluating DeepSeek"):
        single_result, was_skipped = process_single_qa_pair(qa_pair, client, model_name, processed_ids)
        
        if single_result is not None:
            results.append(single_result)
            save_results(results, output_file)
            calculate_and_print_progress(results, len(filtered_qa_pairs))
    
    print_final_results(results, max_duration, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DeepSeek model on video QA dataset")
    parser.add_argument("--data_json_file", type=str, required=True,
                       help="Path to the data JSON file")
    parser.add_argument("--video_dir", type=str, required=True,
                       help="Directory containing video files")
    parser.add_argument("--output_file", type=str,
                       default="./eval_results/deepseek_output.json",
                       help="Output file path for evaluation results")
    parser.add_argument("--max_duration", type=int, default=6000,
                       help="Maximum video duration in seconds")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True) if os.path.dirname(args.output_file) else None
    
    run_deepseek_evaluation(args.data_json_file, args.video_dir, args.output_file, args.max_duration)
