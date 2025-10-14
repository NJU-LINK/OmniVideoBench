import os
import json
import math
import tempfile
from typing import List, Dict, Any
import gc
import psutil
import time 

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

# # 设置环境变量
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
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

# 添加问题视频黑名单
PROBLEMATIC_VIDEOS = set()

def get_resume_unique_id(item):
    """
    根据 video 和 options[0] 为问题生成一个唯一的、可用于断点续传的ID。
    """
    video_name = os.path.basename(item.get("video_path", item.get("video", "")))
    options = item.get("options")
    
    if not video_name or not isinstance(options, list) or len(options) == 0:
        return None
    
    first_option = options[0]
    return f"{video_name}::{first_option}"


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

# ★★★ 优化点 1: 使用 try...finally 确保 video.close() 总能被调用 ★★★
def get_video_chunk_content(video_path, flatten=False):
    """改进的视频内容提取函数，确保资源被释放"""
    video = None  # 先声明
    try:
        video = VideoFileClip(video_path)
        print('video_duration:', video.duration)
        
        if video.audio is None:
            print(f"警告: 视频 {video_path} 没有音频轨道")
            sr = 16000
            audio_np = np.zeros(int(video.duration * sr), dtype=np.float32)
        else:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                    video.audio.write_audiofile(temp_audio_file.name, codec="pcm_s16le", fps=16000, logger=None)
                    audio_np, sr = librosa.load(temp_audio_file.name, sr=16000, mono=True)
            except Exception as e:
                print(f"音频提取失败: {e}，使用静默音频")
                sr = 16000
                audio_np = np.zeros(int(video.duration * sr), dtype=np.float32)
        
        expected_audio_length = int(video.duration * sr)
        if len(audio_np) != expected_audio_length:
            print(f"音频长度调整: {len(audio_np)} -> {expected_audio_length}")
            if len(audio_np) < expected_audio_length:
                audio_np = np.pad(audio_np, (0, expected_audio_length - len(audio_np)), 'constant')
            else:
                audio_np = audio_np[:expected_audio_length]
        
        num_units = math.ceil(video.duration)
        contents = []
        for i in range(num_units):
            start_idx = sr * i
            end_idx = min(sr * (i + 1), len(audio_np))
            if start_idx >= end_idx: continue

            frame_time = min(i + 0.5, video.duration - 0.001) # Use mid-point of segment for frame
            frame = video.get_frame(frame_time)
            image = Image.fromarray(frame.astype(np.uint8))
            
            audio = audio_np[start_idx:end_idx]
            if len(audio) < sr:
                audio = np.pad(audio, (0, sr - len(audio)), 'constant')

            if flatten:
                contents.extend(["<unit>", image, audio.astype(np.float32)])
            else:
                contents.append(["<unit>", image, audio.astype(np.float32)])

        print(f"总共处理了 {len(contents)} 个有效段")
        return contents
    finally:
        if video:
            video.close()

# ★★★ 文件写入函数保持原样，未作改动 ★★★
def write_result_to_file(result, output_file):
    """将单个结果写入JSON文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    existing_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is empty or corrupted, start with an empty list
            existing_results = []
    
    existing_results.append(result)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)


def evaluate_model(dataloader, model, tokenizer, ref_audio_path, output_file, max_duration):
    """模型评估函数，应用了内存和资源管理的优化"""
    ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
    sys_msg = model.get_sys_prompt(ref_audio=ref_audio, mode='omni')

    qa_pairs = filter_qa_pairs_by_duration(dataloader.get_all_qa_pairs(), max_duration)

    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            for item in existing_results:
                resume_id = get_resume_unique_id(item)
                if resume_id:
                    processed_ids.add(resume_id)
            print(f"检测到已存在的输出文件，已加载 {len(processed_ids)} 个已处理问题的记录。")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"警告：输出文件 {output_file} 存在但无法解析，将重新开始。")
            processed_ids = set()
    
    print(f"开始评估，初始内存使用: {psutil.virtual_memory().percent}%")
    print(f"过滤后的QA对数量: {len(qa_pairs)}")
    torch.cuda.empty_cache()
    gc.collect()

    processed_count = 0
    error_count = 0
    skipped_count = 0

    for idx, item in enumerate(qa_pairs):
        loop_start_time = time.time() # ★★★ 优化点 3: 循环开始计时 ★★★

        resume_id = get_resume_unique_id(item)
        if resume_id in processed_ids:
            skipped_count += 1
            continue
            
        video_path = item['video_path']
        video_name = os.path.basename(video_path)
        
        # 新增条件：如果视频名称包含 "A_002"，则跳过
        if "A_002" in video_name:
            print(f"根据要求跳过视频: {video_name}")
            continue
        
        if "Y_014" in video_name:
            print(f"根据要求跳过视频: {video_name}")
            continue
        

        
        print(f"\n处理第 {idx+1}/{len(qa_pairs)} 个视频: {video_name} (唯一ID: {resume_id})")
        
        if video_path in PROBLEMATIC_VIDEOS:
            print(f"跳过问题视频: {video_name}")
            continue
            
        result = {
            "unique_id": item.get('unique_id', f"video_{idx}"), "video": video_name, "duration": item.get("duration", 0),
            "question": item.get("question", ""), "options": item.get("options",[]),
            "correct_answer": item.get("answer", ""), "model_answer": "", "is_correct": False
        }

        if not os.path.exists(video_path):
            print(f"视频不存在: {video_path}")
            result["model_answer"] = f"Error: 视频文件不存在 - {video_path}"
            write_result_to_file(result, output_file)
            error_count += 1
            continue

        try:
            video_proc_start = time.time()
            contents = get_video_chunk_content(video_path)
            print(f"视频预处理耗时: {time.time() - video_proc_start:.2f} 秒")
            
            if not contents:
                raise ValueError("视频内容提取失败")

            question = item.get("question")
            correct_answer = item.get("answer")
            options_text = "\n".join(item.get("options", []))

            question_prompt = (
                "You are given a video. Based on the content of the video, answer the following question:\n\n"
                f"Question:\n{question}\n\n"
                f"Options:\n{options_text}\n\n"
                "Answer with the option's letter directly(e.g., A, B, C, or D)."
                "If your access to the video content is limited, at least one option that is more likely than the others must be chosen."
                "Mustn't give any other reason for can not choose!"
            )

            contents.append(question_prompt)
            msgs = [sys_msg, {"role": "user", "content": contents}]

            model_infer_start = time.time()
            res = model.chat(
                msgs=msgs, tokenizer=tokenizer, sampling=True, temperature=0.5,
                max_new_tokens=512, omni_input=True, use_tts_template=False,
                generate_audio=False, max_slice_nums=1, use_image_id=False, return_dict=True
            )
            print(f"模型推理耗时: {time.time() - model_infer_start:.2f} 秒")
            
            model_answer = res.get("text", str(res))
            is_correct = (model_answer.strip().upper() == correct_answer.strip().upper())
            result["model_answer"] = model_answer
            result["is_correct"] = is_correct
            processed_count += 1
            
        except Exception as e:
            error_msg = f"Error: {type(e).__name__} - {str(e)}"
            print(f"处理视频 {video_name} 时发生错误: {error_msg}")
            result["model_answer"] = error_msg
            result["is_correct"] = False
            PROBLEMATIC_VIDEOS.add(video_path)
            error_count += 1
            if 'cuda' in str(e).lower() or 'memory' in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
        
        io_start_time = time.time()
        write_result_to_file(result, output_file)
        print(f"文件写入耗时: {time.time() - io_start_time:.2f} 秒")
        
        # ★★★ 优化点 2: 显式删除大对象以帮助垃圾回收 ★★★
        if 'contents' in locals(): del contents
        if 'msgs' in locals(): del msgs
        if 'res' in locals(): del res
        
        print(f"结果已写入: {result['unique_id']} - {result['model_answer'][:50]}...")
        
        if (idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"第{idx+1}个视频处理后定期清理内存，当前内存使用: {psutil.virtual_memory().percent}%")
        
        print(f"--- 单个视频总耗时: {time.time() - loop_start_time:.2f} 秒 ---")
    
    print(f"\n评估完成!")
    print(f"本次运行成功处理: {processed_count} 个新视频")
    print(f"本次运行错误数量: {error_count} 个")
    print(f"总共跳过: {skipped_count} 个已处理的视频")

if __name__ == "__main__":
    data_json_file = "/root/siton-tmp/tmp/test.json"
    video_dir = "/root/bayes-gpfs-be2d1a0d1ae94dd3a90607e6f44ad474/videos/omni_videos_v4"
    ref_audio_path = "/root/siton-tmp/code/asr/wav/A_001.wav"
    output_file = "/root/siton-tmp/tmp/test_minicpmo.log"

    dataloader = VideoQADaloader(data_json_file, video_dir)
    model_name = "/root/siton-tmp/models/"
    os.environ['TRANSFORMERS_CACHE'] = model_name
    max_duration = 6000

    print("正在加载模型...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=True,
        local_files_only=True
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("模型加载完成。")

    evaluate_model(dataloader, model, tokenizer, ref_audio_path, output_file, max_duration)

    print(f"\n评测完成，结果已实时保存到 {output_file}")