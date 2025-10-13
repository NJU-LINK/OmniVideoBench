import torch
import json
import os
import string
import re
import traceback
# from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import random
import numpy as np
from accelerate import Accelerator


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#! 请将该函数放到对应模型的eval.py中，不要在该文件中出现
# def load_model_and_processor(model_name: str):
#     """Load and initialize the Qwen model and processor."""
#     print("Loading model and processor...")
#     _device_map = {
#     'thinker.audio_tower': 0,
#     'thinker.visual': 0,
#     'thinker.model.embed_tokens': 0,
    
#     'thinker.model.layers.0': 1,
#     'thinker.model.layers.1': 1,
#     'thinker.model.layers.2': 1,
#     'thinker.model.layers.3': 1,

#     'thinker.model.layers.4': 1,
#     'thinker.model.layers.5': 1,
#     'thinker.model.layers.6': 1,
#     'thinker.model.layers.7': 1,
#     'thinker.model.layers.8': 1,
#     'thinker.model.layers.9': 1,
#     'thinker.model.layers.10': 2,
#     'thinker.model.layers.11': 2,
#     'thinker.model.layers.12': 2,
#     'thinker.model.layers.13': 2,
#     'thinker.model.layers.14': 2,
#     'thinker.model.layers.15': 2,

#     'thinker.model.layers.16': 2,
#     'thinker.model.layers.17': 2,
#     'thinker.model.layers.18': 2,
#     'thinker.model.layers.19': 2,
#     'thinker.model.layers.20': 2,
#     'thinker.model.layers.21': 3,
#     'thinker.model.layers.22': 3,
#     'thinker.model.layers.23': 3,
#     'thinker.model.layers.24': 3,
#     'thinker.model.layers.25': 3,
#     'thinker.model.layers.26': 3,
#     'thinker.model.layers.27': 3,

#     'thinker.model.norm': 0,
#     'thinker.model.rotary_emb': 0,
    
#     'thinker.lm_head': 0,
    
#     'talker': 0,
#     'token2wav': 0
# }
#     model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16,
#         #device_map="auto",
#         device_map = _device_map,
#         #attn_implementation="flash_attention_2",
#         # load_in_8bit=True
#     )
#     print(model.hf_device_map)


#     # processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
#     # model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda",attn_implementation="flash_attention_2",)
#     processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

#     return model, processor
    

def clean_text(text):
    """Clean model's answer to compare with the ground truth."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    
    # Extract content from /box{X} format (LaTeX boxed answer)
    box_pattern = r'/box\{([^}]+)\}'
    box_match = re.search(box_pattern, text)
    if box_match:
        text = box_match.group(1)
    
    # Extract the first letter if it's a multiple-choice question format (e.g., "A. ...")
    if text and len(text) > 1 and text[1] == '.':
        text = text[0]
    
    translator = str.maketrans('', '', string.punctuation)
    text_clean = text.translate(translator)
    # Replace multiple spaces with a single space and remove leading/trailing spaces.
    text_clean = ' '.join(text_clean.split())
    return text_clean

# def create_unique_id(qa_pair):
#     """Create a stable unique identifier for each QA pair using filename_firstword_answer."""
#     video_name = os.path.basename(qa_pair.get("video_path", ""))
#     # Remove file extension from video name
#     video_name = os.path.splitext(video_name)[0]
    
#     question = qa_pair.get("question", "")
#     answer = qa_pair.get("answer", "")
    
#     # Get the first word of the question
#     question_words = question.strip().split()
#     first_word = question_words[0] if question_words else "unknown"
    
#     # Clean the first word and answer to make them filename-safe
#     first_word = re.sub(r'[^\w]', '', first_word.lower())
#     answer_clean = re.sub(r'[^\w]', '', answer.lower())
    
#     # Create unique ID: filename_firstword_answer
#     unique_id = f"{video_name}_{first_word}_{answer_clean}"
    
#     return unique_id

def create_unique_id(qa_pair):
    """Creates a unique identifier for a QA pair based on video and question."""
    # print("qa_pair:", qa_pair)
    if qa_pair.get("video_path", None) and qa_pair["video_path"][-4:] == ".mp4":
        video_name = os.path.basename(qa_pair["video_path"])[:-4]
    else:
        video_name = qa_pair["video"]
    # print("video_name:",video_name)
    question = qa_pair["question"]
    return f"{video_name}_{question}"

# def load_existing_results(output_file):
#     """Load existing results to enable resume functionality."""
#     if os.path.exists(output_file):
#         try:
#             with open(output_file, 'r', encoding='utf-8') as f:
#                 existing_results = json.load(f)
#             # Create a set of processed IDs for quick lookup
#             processed_ids = {result.get('unique_id') for result in existing_results if result.get('unique_id')}
#             print(f"Found {len(existing_results)} existing results, {len(processed_ids)} with valid IDs")
#             return existing_results, processed_ids
#         except Exception as e:
#             print(f"Error loading existing results: {e}")
#             return [], set()
#     return [], set()
def load_existing_results(output_file):
    """
    Loads existing results from a JSON file to allow for resuming an evaluation.

    Args:
        output_file (str): The path to the JSON file containing results.

    Returns:
        tuple: A tuple containing:
            - list: A list of result dictionaries that have already been processed.
            - set: A set of unique IDs for the processed items for quick look-up.
    """
    if not os.path.exists(output_file):
        print("No existing results file found. Starting a new evaluation.")
        return [], set()

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            # Handle empty file case
            content = f.read()
            if not content:
                print("Existing results file is empty. Starting a new evaluation.")
                return [], set()
            results = json.loads(content)
        
        # Create a set of IDs from the loaded results for efficient checking
        processed_ids = {create_unique_id(res) for res in results}
        print(f"Successfully loaded {len(results)} existing results from {output_file}.")
        return results, processed_ids
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading or parsing {output_file}: {e}. Starting a new evaluation.")
        # In case of corruption, start fresh to avoid errors.
        return [], set()

def save_results(results, output_file):
    """Save results to file with error handling."""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

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

def validate_qa_pair(qa_pair):
    """Validate if QA pair has all required fields."""
    video_path = qa_pair.get("video_path")
    question = qa_pair.get("question")
    options = qa_pair.get("options")
    correct_answer = qa_pair.get("answer")
    
    return all([video_path, question, options, correct_answer])


def extract_model_answer(response_text,prompt=None):
    """Extract and clean model answer from generated text."""
    # Extract the model's answer from the response
    if "assistant" in response_text:
        model_answer = response_text.split("assistant")[-1].strip()
    else:
        # If no "assistant" marker, take the last part after the prompt
        model_answer = response_text.split(prompt)[-1].strip()
    
    # Additional processing for common answer formats
    # Handle /box{X} format (LaTeX boxed answers)
    box_pattern = r'/box\{([^}]+)\}'
    box_match = re.search(box_pattern, model_answer)
    if box_match:
        model_answer = box_match.group(1).strip()
    
    # Handle \boxed{X} format (another LaTeX variant)
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, model_answer)
    if boxed_match:
        model_answer = boxed_match.group(1).strip()
    
    return model_answer


def extract_model_answer_thinking(response_text, prompt=None):
    """Extract and clean model answer from thinking model's generated text.
    
    Thinking models output content in format:
    <think>thinking process</think>
    Final answer
    
    This function removes the <thinking> section and extracts only the final answer.
    """
    # First, remove the <thinking> section if it exists
    thinking_pattern = r'<think>.*?</think>'
    # Remove the thinking section and any surrounding whitespace
    cleaned_response = re.sub(thinking_pattern, '', response_text, flags=re.DOTALL).strip()
    
    # If the response is empty after removing thinking section, use original response
    if not cleaned_response:
        cleaned_response = response_text
    
    # Now apply the standard extraction logic from extract_model_answer
    if "assistant" in cleaned_response:
        model_answer = cleaned_response.split("assistant")[-1].strip()
    else:
        # If no "assistant" marker, take the last part after the prompt
        model_answer = cleaned_response.split(prompt)[-1].strip() if prompt else cleaned_response
    
    # Additional processing for common answer formats
    # Handle /box{X} format (LaTeX boxed answers)
    box_pattern = r'/box\{([^}]+)\}'
    box_match = re.search(box_pattern, model_answer)
    if box_match:
        model_answer = box_match.group(1).strip()
    
    # Handle \boxed{X} format (another LaTeX variant)
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, model_answer)
    if boxed_match:
        model_answer = boxed_match.group(1).strip()
    
    return model_answer


def handle_processing_error(e, video_path):
    """Handle and log processing errors."""
    print(f"An error occurred during model inference for {video_path}:")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {str(e)}")
    print(f"Full Traceback:\n{traceback.format_exc()}")
    return f"Error: {type(e).__name__} - {str(e)}"