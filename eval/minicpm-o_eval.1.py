import os
import json
import math
import tempfile
from typing import List, Dict, Any
import gc
import psutil

import numpy as np
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
import librosa
import soundfile as sf
import torch
from transformers import AutoModel, AutoTokenizer
import random

import sys
sys.path.append("/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench")
sys.path.append('/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/models/MiniCPM-o-2_6')


from dataloader import VideoQADaloader

# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


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

# ★★★ 新增：用于断点续传的唯一ID生成函数 ★★★
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

def get_video_chunk_content(video_path, flatten=False):
    """改进的视频内容提取函数，确保音频索引一致性"""
    video = VideoFileClip(video_path)
    print('video_duration:', video.duration)
    
    if video.audio is None:
        print(f"警告: 视频 {video_path} 没有音频轨道")
        duration = video.duration
        sr = 16000
        audio_np = np.zeros(int(duration * sr), dtype=np.float32)
    else:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                temp_audio_file_path = temp_audio_file.name
                video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
                audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
        except Exception as e:
            print(f"音频提取失败: {e}，使用静默音频")
            duration = video.duration
            sr = 16000
            audio_np = np.zeros(int(duration * sr), dtype=np.float32)
    
    expected_audio_length = int(video.duration * sr)
    if len(audio_np) != expected_audio_length:
        print(f"音频长度调整: {len(audio_np)} -> {expected_audio_length}")
        if len(audio_np) < expected_audio_length:
            audio_np = np.pad(audio_np, (0, expected_audio_length - len(audio_np)), mode='constant', constant_values=0)
        else:
            audio_np = audio_np[:expected_audio_length]
    
    num_units = math.ceil(video.duration)
    
    valid_segments = []
    for i in range(num_units):
        start_idx = sr * i
        end_idx = sr * (i + 1)
        
        if start_idx >= len(audio_np):
            print(f"警告: 第{i}段音频开始索引超出范围，跳过此段")
            continue
            
        if end_idx > len(audio_np):
            available_audio = audio_np[start_idx:]
            if len(available_audio) < sr * 0.1:
                print(f"警告: 第{i}段剩余音频太短({len(available_audio)}样本)，跳过此段")
                continue
            end_idx = len(audio_np)
        
        valid_segments.append((i, start_idx, end_idx))
    
    contents = []
    for segment_idx, start_idx, end_idx in valid_segments:
        try:
            frame_time = min(segment_idx + 1, video.duration - 0.001)
            frame = video.get_frame(frame_time)
            image = Image.fromarray((frame).astype(np.uint8))
            
            audio = audio_np[start_idx:end_idx]
            
            if len(audio) < sr and segment_idx < num_units - 1:
                audio = np.pad(audio, (0, sr - len(audio)), mode='constant', constant_values=0)
            elif len(audio) > sr:
                audio = audio[:sr]
            
            audio = audio.astype(np.float32)
            
            if flatten:
                contents.extend(["<unit>", image, audio])
            else:
                contents.append(["<unit>", image, audio])
                
        except Exception as e:
            print(f"处理第{segment_idx}段时出错: {e}")
            video.close()
            raise e
    
    print(f"总共处理了 {len(valid_segments)} 个有效音频段，共 {num_units} 段")
    
    video.close()
    return contents

def write_result_to_file(result, output_file):
    """将单个结果写入JSON文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
    
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        existing_results = []
    
    existing_results.append(result)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

def evaluate_model(dataloader, model, tokenizer, ref_audio_path, output_file, max_duration):
    """改进的模型评估函数，支持断点续传并实时写入结果"""
    ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
    sys_msg = model.get_sys_prompt(ref_audio=ref_audio, mode='omni')

    qa_pairs = filter_qa_pairs_by_duration(dataloader.get_all_qa_pairs(), max_duration)

    # ★★★ 新增：断点续传逻辑 ★★★
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
        # ★★★ 新增：检查是否已处理 ★★★
        resume_id = get_resume_unique_id(item)
        if resume_id in processed_ids:
            skipped_count += 1
            if (idx + 1) % 100 == 0: # 每隔一段时间打印一次跳过信息，避免刷屏
                print(f"已跳过 {skipped_count} 个已处理的问题...")
            continue
            
        video_path = item['video_path']
        unique_id = item.get('unique_id', f"video_{idx}")
        video_name = os.path.basename(video_path)
        
        print(f"\n处理第 {idx+1}/{len(qa_pairs)} 个视频: {video_name} (唯一ID: {resume_id})")
        
        # -------------------- 新增：跳过 Z_043.mp4 --------------------
        if video_name == "Z_043.mp4":
            print(f"手动跳过视频: {video_name}")
            continue
        # -------------------------------------------------------------
        if video_path in PROBLEMATIC_VIDEOS:
            print(f"跳过问题视频: {video_name}")
            continue
            
        result = {
            "unique_id": unique_id,
            "video": video_name,
            "duration": item.get("duration", 0),
            "question": item.get("question", ""),
            "options": item.get("options",[]),
            "correct_answer": item.get("answer", ""),
            "model_answer": "",
            "is_correct": False
        }

        options = item.get("options", [])
        options_text = "\n".join(options)

        if not os.path.exists(video_path):
            print(f"视频不存在: {video_path}")
            result["model_answer"] = f"Error: 视频文件不存在 - {video_path}"
            write_result_to_file(result, output_file)
            error_count += 1
            continue

        try:
            contents = get_video_chunk_content(video_path)
            
            if not contents:
                print(f"视频内容提取失败，跳过: {video_name}")
                result["model_answer"] = "Error: 视频内容提取失败"
                write_result_to_file(result, output_file)
                PROBLEMATIC_VIDEOS.add(video_path)
                error_count += 1
                continue
            
            image_count = sum(1 for item in contents if isinstance(item[1], Image.Image))
            audio_count = sum(1 for item in contents if isinstance(item[2], np.ndarray))
            
            print(f"内容验证: 图像数量={image_count}, 音频数量={audio_count}")
            
            if image_count != audio_count:
                print(f"警告: 图像和音频数量不匹配 ({image_count} vs {audio_count})，跳过此视频")
                result["model_answer"] = f"Error: 图像和音频数量不匹配 ({image_count} vs {audio_count})"
                write_result_to_file(result, output_file)
                PROBLEMATIC_VIDEOS.add(video_path)
                error_count += 1
                continue

            question = item.get("question")
            correct_answer = item.get("answer")

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

            try:
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
                
                model_answer = res["text"] if isinstance(res, dict) and "text" in res else str(res)
                is_correct = (model_answer.strip().upper() == correct_answer.strip().upper())

                result["model_answer"] = model_answer
                result["is_correct"] = is_correct
                processed_count += 1
                
            except torch.cuda.OutOfMemoryError as oom_error:
                error_msg = f"Error: OutOfMemoryError - CUDA out of memory. {str(oom_error)}"
                print(f"CUDA内存不足: {oom_error}")
                result["model_answer"] = error_msg
                result["is_correct"] = False
                error_count += 1
                torch.cuda.empty_cache()
                gc.collect()
                
            except AssertionError as ae:
                error_msg = f"Error: AssertionError - 音频索引不匹配: {str(ae)}"
                print(f"断言错误: {ae}")
                result["model_answer"] = error_msg
                result["is_correct"] = False
                PROBLEMATIC_VIDEOS.add(video_path)
                error_count += 1
                
            except Exception as model_error:
                error_msg = f"Error: {type(model_error).__name__} - {str(model_error)}"
                print(f"模型推理错误: {model_error}")
                result["model_answer"] = error_msg
                result["is_correct"] = False
                PROBLEMATIC_VIDEOS.add(video_path)
                error_count += 1

        except Exception as e:
            error_msg = f"Error: {type(e).__name__} - {str(e)}"
            print(f"处理视频 {video_name} 时发生错误: {e}")
            result["model_answer"] = error_msg
            result["is_correct"] = False
            PROBLEMATIC_VIDEOS.add(video_path)
            error_count += 1
        
        write_result_to_file(result, output_file)
        print(f"结果已写入: {result['unique_id']} - {result['model_answer'][:50]}...")
        
        if (idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            memory_percent = psutil.virtual_memory().percent
            print(f"第{idx+1}个视频处理完成，当前内存使用: {memory_percent}%")
            
            if memory_percent > 85:
                print("内存使用过高，执行强制清理...")
                torch.cuda.empty_cache()
                gc.collect()
    
    print(f"\n评估完成!")
    print(f"本次运行成功处理: {processed_count} 个新视频")
    print(f"本次运行错误数量: {error_count} 个")
    print(f"总共跳过: {skipped_count} 个已处理的视频")
    print(f"问题视频数量: {len(PROBLEMATIC_VIDEOS)}")

if __name__ == "__main__":
    #data_json_file = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/final_data/final_data copy.json"
    data_json_file = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/final_data/qa_data.json"
    video_dir = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/omni_videos_v2"
    ref_audio_path = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/models/MiniCPM-o-2_6/assets/demo.wav"

    #output_file = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/model/eval_results/minicpmo_final_out.json"
    output_file = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/model/minicpmo_final_out.json"

    dataloader = VideoQADaloader(data_json_file, video_dir)
    model_name = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/models/MiniCPM-o-2_6"
    os.environ['TRANSFORMERS_CACHE'] = model_name

    max_duration = 6000

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=True,
        local_files_only=True
    )

    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    evaluate_model(dataloader, model, tokenizer, ref_audio_path, output_file, max_duration)

    print(f"\n评测完成，结果已实时保存到 {output_file}")