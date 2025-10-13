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
import sys
import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer

from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init

# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


sys.path.append("/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench")
from dataloader import VideoQADaloader

sys.path.append("/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench")
from utils.utils import set_seed, filter_qa_pairs_by_duration

set_seed(42)


# 添加问题视频黑名单
PROBLEMATIC_VIDEOS = set()

def get_video_chunk_content(video_path, flatten=False):
    """改进的视频内容提取函数，确保音频索引一致性"""
    video = VideoFileClip(video_path)
    print('video_duration:', video.duration)
    
    # 检查视频是否有音频轨道
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
    
    # 确保音频长度与视频时长匹配
    expected_audio_length = int(video.duration * sr)
    if len(audio_np) != expected_audio_length:
        print(f"音频长度调整: {len(audio_np)} -> {expected_audio_length}")
        if len(audio_np) < expected_audio_length:
            audio_np = np.pad(audio_np, (0, expected_audio_length - len(audio_np)), mode='constant', constant_values=0)
        else:
            audio_np = audio_np[:expected_audio_length]
    
    num_units = math.ceil(video.duration)
    
    # 预先计算所有有效的音频段，确保索引一致性
    valid_segments = []
    for i in range(num_units):
        start_idx = sr * i
        end_idx = sr * (i + 1)
        
        # 检查边界条件
        if start_idx >= len(audio_np):
            print(f"警告: 第{i}段音频开始索引超出范围，跳过此段")
            continue
            
        if end_idx > len(audio_np):
            available_audio = audio_np[start_idx:]
            if len(available_audio) < sr * 0.1:  # 如果剩余音频太短，跳过
                print(f"警告: 第{i}段剩余音频太短({len(available_audio)}样本)，跳过此段")
                continue
            # 对于最后一段，调整end_idx到实际音频长度
            end_idx = len(audio_np)
        
        valid_segments.append((i, start_idx, end_idx))
    
    # 基于有效段生成内容
    contents = []
    for segment_idx, start_idx, end_idx in valid_segments:
        try:
            # 获取帧
            frame_time = min(segment_idx + 1, video.duration - 0.001)
            frame = video.get_frame(frame_time)
            image = Image.fromarray((frame).astype(np.uint8))
            
            # 提取音频段
            audio = audio_np[start_idx:end_idx]
            
            # 确保音频长度为1秒（16000样本），除了最后一段
            if len(audio) < sr and segment_idx < num_units - 1:
                audio = np.pad(audio, (0, sr - len(audio)), mode='constant', constant_values=0)
            elif len(audio) > sr:
                audio = audio[:sr]
            
            # 确保数据类型
            audio = audio.astype(np.float32)
            
            if flatten:
                contents.extend(["<unit>", image, audio])
            else:
                contents.append(["<unit>", image, audio])
                
        except Exception as e:
            print(f"处理第{segment_idx}段时出错: {e}")
            # 如果单个段处理失败，整个视频标记为有问题
            video.close()
            raise e
    
    print(f"总共处理了 {len(valid_segments)} 个有效音频段，共 {num_units} 段")
    
    video.close()
    return contents

def write_result_to_file(result, output_file):
    """将单个结果写入JSON文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 如果文件不存在，创建空列表
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
    
    # 读取现有结果
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        existing_results = []
    
    # 添加新结果
    existing_results.append(result)
    
    # 写回文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

def evaluate_model(dataloader, model, tokenizer, output_file,max_duration):
    """改进的模型评估函数，实时写入结果"""

    # 使用filter_qa_pairs_by_duration过滤QA对
    qa_pairs = filter_qa_pairs_by_duration(dataloader.get_all_qa_pairs(),max_duration)
    
    # 内存监控
    print(f"开始评估，初始内存使用: {psutil.virtual_memory().percent}%")
    print(f"过滤后的QA对数量: {len(qa_pairs)}")
    torch.cuda.empty_cache()
    gc.collect()

    processed_count = 0
    error_count = 0

    # 回答正确的问题数目
    right_count = 0

    for idx, item in enumerate(qa_pairs):
        video_path = item['video_path']
        unique_id = item.get('unique_id', f"video_{idx}")
        video_name = os.path.basename(video_path)
        
        print(f"\n处理第 {idx+1}/{len(qa_pairs)} 个视频: {video_name}")
        
        # 检查是否在黑名单中
        if video_path in PROBLEMATIC_VIDEOS:
            print(f"跳过问题视频: {video_name}")
            continue
            
        # 初始化结果对象
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

        # 处理选项格式
        options = item.get("options", [])
        options_text = "\n".join(options)

        if not os.path.exists(video_path):
            print(f"视频不存在: {video_path}")
            result["model_answer"] = f"Error: 视频文件不存在 - {video_path}"
            write_result_to_file(result, output_file)
            error_count += 1
            continue

        try:
            # --- HumanOmni 数据预处理 ---
            # 使用HumanOmni的processor来处理视频和音频
            # processor['video'] 和 processor['audio'] 直接接受文件路径
            video_tensor = processor['video'](video_path)
            audio_tensor = processor['audio'](video_path)[0] # processor['audio'] 返回 (tensor, sample_rate)

            # --- 构建Prompt ---
            question = item.get("question")
            options = item.get("options", [])
            correct_answer = item.get("answer")
            options_text = "\n".join(options)

            instruct = (
                "You are given a video. Based on the content of the video, answer the following question:\n\n"
                f"Question:\n{question}\n\n"
                f"Options:\n{options_text}\n\n"
                "Answer with the option's letter directly(e.g., A, B, C, or D)."
                "If your access to the video content is limited, at least one option that is more likely than the others must be chosen."
                "Mustn't give any other reason for can not choose!"
            )

            # ----Human Omni模型推理
            try:
                output = mm_infer(
                    image_or_video=video_tensor,
                    instruct=instruct,
                    model=model,
                    tokenizer=tokenizer,
                    audio=audio_tensor,
                    modal='video_audio',
                    question=question, # mm_infer 需要 question 参数
                    bert_tokeni=bert_tokenizer,
                    do_sample=False,
                )
                
                model_answer = output.strip()
                is_correct = (model_answer.upper() == str(correct_answer).strip().upper())

                result["model_answer"] = model_answer
                result["is_correct"] = is_correct

                if is_correct == True:
                    right_count += 1

                processed_count += 1                
                
            except torch.cuda.OutOfMemoryError as oom_error:
                error_msg = f"Error: OutOfMemoryError - CUDA out of memory. {str(oom_error)}"
                print(f"CUDA内存不足: {oom_error}")
                result["model_answer"] = error_msg
                result["is_correct"] = False
                error_count += 1
                
                # 清理内存并继续
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
        
        # 写入结果到文件
        write_result_to_file(result, output_file)
        print(f"结果已写入: {result['unique_id']} - {result['model_answer'][:50]}...")
        
        # 定期内存清理
        if (idx + 1) % 5 == 0:  # 更频繁的清理
            torch.cuda.empty_cache()
            gc.collect()
            memory_percent = psutil.virtual_memory().percent
            print(f"第{idx+1}个视频处理完成，当前内存使用: {memory_percent}%")
            
            # 如果内存使用过高，强制清理
            if memory_percent > 85:
                print("内存使用过高，执行强制清理...")
                torch.cuda.empty_cache()
                gc.collect()
    
    print(f"\n评估完成!")
    print(f"成功处理: {processed_count} 个视频")
    print(f"错误数量: {error_count} 个")
    print(f"问题视频数量: {len(PROBLEMATIC_VIDEOS)}")
    print(f"正确率为：{( right_count / processed_count ) * 100: .2f}%")

if __name__ == "__main__":
    # task_name = "first_test"
    task_name = "qa_data"
    # data_json_file = "/fs-computility/llm_code_collab/liujiaheng/caoruili/omni-bench/data/merged_qas_1_0811.json"
    data_json_file = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/final_data/qa_data.json"
    video_dir = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/omni_videos_v2"
    # ref_audio_path = "/fs-computility/llm_code_collab/liujiaheng/caoruili/models/MiniCPM-o-2_6/assets/demo.wav"
    model_name = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/models/HumanOmni-7B"
    model_basename = os.path.basename(model_name)
    output_file = f"/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/eval_results/{model_basename}_{task_name}.json"

    dataloader = VideoQADaloader(data_json_file, video_dir)

    max_duration=6000

    # 初始化BERT分词器
    bert_model = "./bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model,
                                        cache_dir=bert_model,
                                        local_files_only=True)
    print("init tokenizer")
    # 禁用Torch初始化
    disable_torch_init()

    # 初始化模型、处理器和分词器
    model, processor, tokenizer = model_init(model_path=model_name)
    model = model.eval().cuda()
    print(model.hf_device_map)

    # 直接在evaluate_model中处理结果写入
    evaluate_model(dataloader, model, tokenizer, output_file,max_duration)

    print(f"\n评测完成，结果已实时保存到 {output_file}")
