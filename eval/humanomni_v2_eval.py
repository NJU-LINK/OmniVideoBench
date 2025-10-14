

import torch
import json
import os
import string
import re
import warnings

from tqdm import tqdm
import soundfile as sf
# from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import sys

# 抑制不必要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="qwen_omni_utils")
warnings.filterwarnings("ignore", category=UserWarning, module="qwen_vl_utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

# sys.path.append("../../")
sys.path.append("/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench")
from dataloader import VideoQADaloader


# sys.path.append("../../utils/")
sys.path.append("/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/utils")

from utils.utils import (
    # load_model_and_processor,
    clean_text,
    save_results,
    filter_qa_pairs_by_duration,
    validate_qa_pair,
    extract_model_answer,
    handle_processing_error,
    set_seed
    )
# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
    """Extracts content from the <answer> tag."""
    match = re.search(r'<answer>(.*)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback for answers without tags
    # Split by common delimiters and check for valid options
    parts = re.split(r'[:\s\n]', text)
    potential_answers = [p.strip() for p in parts if p.strip() and len(p.strip()) == 1 and p.strip().isalpha()]
    
    if potential_answers:
        return potential_answers[-1]  # Return the last likely option
        
    return text.strip()

def build_conversation(video_path, question, options):
    """Build conversation format for the model."""
    options_text = "\n".join(options)
    prompt = (
        "You are given a video. Based on the content of the video, answer the following question:\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Provide your final answer between the <answer> </answer> tags. "
        "Please provide only the single option letter (e.g., A, B, C, D, etc.)."
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


def process_multimedia_input(conversation, processor):
    """Process multimedia information from conversation."""
    USE_AUDIO_IN_VIDEO = True
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    print("processing video...")
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    # TODO: 这里按照单卡上限对长视频进行采样，保持帧数不超过120
    # MAX_FRAMES=120
    MAX_FRAMES = 32
    total_frames = len(videos[0])
    print("total_frames:", total_frames)
    if total_frames > MAX_FRAMES:
        # 均匀采样
        step = total_frames / MAX_FRAMES
        sampled_indices = [int(i * step) for i in range(MAX_FRAMES)]
        videos = [videos[0][i] for i in sampled_indices]
    print("frames of videos:", len(videos))
    inputs = processor(
        text=text, 
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt", 
        padding=True, 
        use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    
    # 检查输入token数量
    if hasattr(inputs, 'input_ids') and inputs.input_ids is not None:
        token_count = inputs.input_ids.shape[-1]
        print(f"Input token count: {token_count}")
        if token_count > 30000:  # 接近32768限制时警告
            print(f"WARNING: Token count ({token_count}) is approaching model limit (32768)")
    
    return inputs, USE_AUDIO_IN_VIDEO


def generate_model_response(model, inputs, use_audio_in_video):
    """Generate response from the model."""
    inputs = inputs.to(model.device).to(model.dtype)
    
    print("shape of inputs:", inputs.input_ids.shape)
    # 保存输入长度用于后续截断生成的tokens
    input_length = inputs.input_ids.shape[-1] if hasattr(inputs, 'input_ids') and inputs.input_ids is not None else 0
    
    print("start inferencing...")
    try:
        with torch.no_grad():
            generation_output = model.generate(
                **inputs, 
                use_audio_in_video=use_audio_in_video,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True
            )
            
            # Handle different return types from model.generate()
            if isinstance(generation_output, tuple) and len(generation_output) == 2:
                text_ids, audio = generation_output
            else:
                # If only text_ids is returned, set audio to None
                text_ids = generation_output
                audio = None
                
    except Exception as e:
        print(f"Error during generation: {e}")
        # 如果生成失败，返回错误信息
        raise e
    
    del inputs
    return text_ids, audio, input_length



def calculate_and_print_progress(results, total_items, max_duration):
    """Calculate and print current progress."""
    if not results:
        return
        
    current_correct = sum(1 for r in results if r.get('is_correct', False))
    failed_videos = sum(1 for r in results if 'error' in r.get('model_answer', '').lower() or 
                       'not found' in r.get('model_answer', '').lower())
    successful_videos = len(results) - failed_videos
    current_accuracy = current_correct / successful_videos if successful_videos > 0 else 0
    
    print(f"Progress: {len(results)}/{total_items}")
    print(f"  ✅ Successful: {successful_videos}, ❌ Failed: {failed_videos}")
    print(f"  🎯 Accuracy: {current_accuracy:.2%} ({current_correct}/{successful_videos})")


def print_final_results(results, max_duration, output_file):
    """Print final evaluation results."""
    total_predictions = len(results)
    if total_predictions > 0:
        final_correct = sum(1 for r in results if r.get('is_correct', False))
        failed_videos = sum(1 for r in results if 'error' in r.get('model_answer', '').lower() or 
                           'not found' in r.get('model_answer', '').lower())
        successful_videos = total_predictions - failed_videos
        accuracy = final_correct / successful_videos if successful_videos > 0 else 0
        
        print(f"\n🎉 Evaluation completed!")
        print(f"📊 Total questions processed (duration < {max_duration}s): {total_predictions}")
        print(f"✅ Successful processing: {successful_videos}")
        print(f"❌ Failed processing: {failed_videos}")
        print(f"🎯 Correct predictions: {final_correct}")
        print(f"📈 Accuracy: {accuracy:.2%} (based on successfully processed videos)")
        print(f"💾 Detailed results saved to: {output_file}")
    else:
        print("No questions were processed.")


def process_single_qa_pair(qa_pair, model, processor):
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
        
        # Build conversation and process multimedia
        conversation, prompt = build_conversation(video_path, question, options)
        inputs, use_audio_in_video = process_multimedia_input(conversation, processor)
        
        # Generate model response
        text_ids, audio, input_length = generate_model_response(model, inputs, use_audio_in_video)
        
        # Extract and evaluate answer
        print("decode the response...")
        # 只解码新生成的tokens，排除输入部分
        generated_tokens = text_ids[:, input_length:] if text_ids.shape[-1] > input_length else text_ids
        response_text = processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        model_answer = extract_answer(response_text)
        single_result['model_answer'] = model_answer
        single_result['is_correct'] = clean_text(model_answer) == clean_text(correct_answer)
        
    except Exception as e:
        # 特殊处理视频解码相关的错误
        error_str = str(e)
        if any(keyword in error_str.lower() for keyword in ['nal unit', 'invalid data', 'decord', 'avcodec', 'video']):
            print(f"❌ Video decoding failed for {video_path}: {type(e).__name__}: {e}")
            
            # 标记不同类型的解码错误
            if 'nal unit' in error_str.lower():
                single_result['model_answer'] = 'Video corrupted: NAL unit error - unable to decode frames'
            elif 'invalid data' in error_str.lower():
                single_result['model_answer'] = 'Video corrupted: Invalid data format'
            else:
                single_result['model_answer'] = f'Video decoding error: {type(e).__name__}'
                
            # 在结果中添加错误详情以便后续分析
            single_result['error_type'] = 'video_decode_failure'
            single_result['error_details'] = str(e)[:200]  # 截断错误信息
        else:
            print(f"❌ Processing error for {video_path}: {type(e).__name__}: {e}")
            single_result['model_answer'] = handle_processing_error(e, video_path)
    
    return single_result

def load_model_and_processor(model_name: str):
    """Load and initialize the Qwen model and processor."""
    print("Loading model and processor...")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        # load_in_8bit=True
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    return model, processor

def run_qwen_evaluation(model_name:str, data_json_file: str=None, video_dir: str=None, output_file: str=None, max_duration: int=None):
    """Run the evaluation on the dataset using the Qwen-Omni model."""
    set_seed(42)

    # Initialize model and processor
    # MODEL_NAME = "/fs-computility/llm_code_collab/liujiaheng/shihao/OmniBench-Video/Qwen/Qwen2.5-Omni-7B"
    
    model, processor = load_model_and_processor( model_name )
    
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
    
    # Initialize empty results list
    results = []
    
    print(f"Starting evaluation on {len(filtered_qa_pairs)} QA pairs with {model_name}...")
    
    # Process each QA pair
    for qa_pair in tqdm(filtered_qa_pairs, desc="Evaluating Qwen-Omni"):
        single_result = process_single_qa_pair(qa_pair, model, processor)
        
        if single_result is not None:
            results.append(single_result)
            
            # Save results after each processing
            save_results(results, output_file)
            
            # Print current progress
            calculate_and_print_progress(results, len(filtered_qa_pairs), max_duration)
    
    # Print final results
    print_final_results(results, max_duration, output_file)


if __name__ == "__main__":
    task_name = "first_test"

    base_dir = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench"
    data_json_file=f"{base_dir}/data/merged_qas_1_0817.json"
    video_dir="/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/omni_videos_v1"
    
    # model_name = f"{base_dir}/models/HumanOmniV2"
    model_name = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/models/HumanOmniV2"
    model_basename = os.path.basename(model_name)
    output_file = f"{base_dir}/eval_results/{model_basename}_{task_name}.json"    

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    max_duration = 6000
    
    run_qwen_evaluation(model_name, data_json_file, video_dir, output_file, max_duration)