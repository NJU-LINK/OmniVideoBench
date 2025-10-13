import os
import json
import math
import tempfile
from typing import List, Dict, Any
import gc
import psutil
import logging  # 1. 添加 logging 模块导入

import numpy as np
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
import librosa
import soundfile as sf
import sys
import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 1. 添加自定义dataloader.py所在的绝对路径（omni-bench根目录），并插入到sys.path最前面
custom_dataloader_path = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/"
sys.path.insert(0, custom_dataloader_path)  # 插入到最前面，优先加载

# 2. 导入自定义的VideoQADaloader（此时会优先加载omni-bench目录下的dataloader.py）
from dataloader import VideoQADaloader
# sys.path.append("../../")
# from dataloader import VideoQADaloader

sys.path.append("../../utils/")
from utils.utils import set_seed, filter_qa_pairs_by_duration

set_seed(42)

# 添加问题视频黑名单
PROBLEMATIC_VIDEOS = set()

def get_video_chunk_content(video_path, flatten=False):
    """改进的视频内容提取函数，确保音频索引一致性"""
    video = VideoFileClip(video_path)
    logging.info(f'video_duration: {video.duration}') # print -> logging.info
    
    # 检查视频是否有音频轨道
    if video.audio is None:
        logging.warning(f"警告: 视频 {video_path} 没有音频轨道") # print -> logging.warning
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
            logging.error(f"音频提取失败: {e}，使用静默音频") # print -> logging.error
            duration = video.duration
            sr = 16000
            audio_np = np.zeros(int(duration * sr), dtype=np.float32)
    
    # 确保音频长度与视频时长匹配
    expected_audio_length = int(video.duration * sr)
    if len(audio_np) != expected_audio_length:
        logging.info(f"音频长度调整: {len(audio_np)} -> {expected_audio_length}") # print -> logging.info
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
            logging.warning(f"警告: 第{i}段音频开始索引超出范围，跳过此段") # print -> logging.warning
            continue
            
        if end_idx > len(audio_np):
            available_audio = audio_np[start_idx:]
            if len(available_audio) < sr * 0.1:  # 如果剩余音频太短，跳过
                logging.warning(f"警告: 第{i}段剩余音频太短({len(available_audio)}样本)，跳过此段") # print -> logging.warning
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
            logging.error(f"处理第{segment_idx}段时出错: {e}") # print -> logging.error
            # 如果单个段处理失败，整个视频标记为有问题
            video.close()
            raise e
    
    logging.info(f"总共处理了 {len(valid_segments)} 个有效音频段，共 {num_units} 段") # print -> logging.info
    
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

def evaluate_model(dataloader, model,processor,  tokenizer, output_file,max_duration):
    """改进的模型评估函数，实时写入结果"""

    # 使用filter_qa_pairs_by_duration过滤QA对
    qa_pairs = filter_qa_pairs_by_duration(dataloader.get_all_qa_pairs(),max_duration)
    
    # 内存监控
    logging.info(f"开始评估，初始内存使用: {psutil.virtual_memory().percent}%") # print -> logging.info
    logging.info(f"过滤后的QA对数量: {len(qa_pairs)}") # print -> logging.info
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
        
        logging.info(f"\n处理第 {idx+1}/{len(qa_pairs)} 个视频: {video_name}") # print -> logging.info
        
        # 检查是否在黑名单中
        if video_path in PROBLEMATIC_VIDEOS:
            logging.warning(f"跳过问题视频: {video_name}") # print -> logging.warning
            continue
            
        # 初始化结果对象
        result = {
            # "unique_id": unique_id, # <--- 第1处修改：注释掉此行
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
            logging.error(f"视频不存在: {video_path}") # print -> logging.error
            result["model_answer"] = f"Error: 视频文件不存在 - {video_path}"
            write_result_to_file(result, output_file)
            error_count += 1
            continue

        try:
            # --- HumanOmni 数据预处理 ---
            # 使用HumanOmni的processor来处理视频和音频
            # processor['video'] 和 processor['audio'] 直接接受文件路径
            video_tensor = processor['video'](video_path)

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

            # ----Human Omni模型推理----
            try:
                output = mm_infer(
                    image_or_video=video_tensor,
                    instruct=instruct,
                    model=model,
                    tokenizer=tokenizer,
                    # audio=audio_tensor,
                    # modal='video_audio',
                    # question=question, # mm_infer 需要 question 参数
                    # bert_tokeni=bert_tokenizer,
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
                logging.error(f"CUDA内存不足: {oom_error}") # print -> logging.error
                result["model_answer"] = error_msg
                result["is_correct"] = False
                error_count += 1
                
                # 清理内存并继续
                torch.cuda.empty_cache()
                gc.collect()
                
            except AssertionError as ae:
                error_msg = f"Error: AssertionError - 音频索引不匹配: {str(ae)}"
                logging.error(f"断言错误: {ae}") # print -> logging.error
                result["model_answer"] = error_msg
                result["is_correct"] = False
                PROBLEMATIC_VIDEOS.add(video_path)
                error_count += 1
                
            except Exception as model_error:
                error_msg = f"Error: {type(model_error).__name__} - {str(model_error)}"
                logging.error(f"模型推理错误: {model_error}") # print -> logging.error
                result["model_answer"] = error_msg
                result["is_correct"] = False
                PROBLEMATIC_VIDEOS.add(video_path)
                error_count += 1

        except Exception as e:
            error_msg = f"Error: {type(e).__name__} - {str(e)}"
            logging.error(f"处理视频 {video_name} 时发生错误: {e}") # print -> logging.error
            result["model_answer"] = error_msg
            result["is_correct"] = False
            PROBLEMATIC_VIDEOS.add(video_path)
            error_count += 1
        
        # 写入结果到文件
        write_result_to_file(result, output_file)
        # <--- 第2处修改：修改下面的print语句，因为它用到了unique_id
        logging.info(f"结果已写入: {video_name} - {result['model_answer'][:50]}...") # print -> logging.info
        
        # 定期内存清理
        if (idx + 1) % 5 == 0:  # 更频繁的清理
            torch.cuda.empty_cache()
            gc.collect()
            memory_percent = psutil.virtual_memory().percent
            logging.info(f"第{idx+1}个视频处理完成，当前内存使用: {memory_percent}%") # print -> logging.info
            
            # 如果内存使用过高，强制清理
            if memory_percent > 85:
                logging.warning("内存使用过高，执行强制清理...") # print -> logging.warning
                torch.cuda.empty_cache()
                gc.collect()
    
    logging.info(f"\n评估完成!") # print -> logging.info
    logging.info(f"成功处理: {processed_count} 个视频") # print -> logging.info
    logging.info(f"错误数量: {error_count} 个") # print -> logging.info
    logging.info(f"问题视频数量: {len(PROBLEMATIC_VIDEOS)}") # print -> logging.info
    logging.info(f"正确率为：{( right_count / processed_count ) * 100: .2f}%") # print -> logging.info

if __name__ == "__main__":
    # 2. 添加此日志配置块
    log_filename = 'VideoLLaMA2_log.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'), # 写入文件
            logging.StreamHandler(sys.stdout)  # 同时输出到控制台
        ]
    )
    # --- 日志配置结束 ---

    task_name = "first_test"
    # task_name = "waiting_hy"
    #data_json_file = "/fs-computility/llm_code_collab/liujiaheng/caoruili/omni-bench/data/merged_qas_1_0811.json"
    data_json_file = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/final_data/qa_data.json"

    
    # data_json_file = "/fs-computility/llm_code_collab/liujiaheng/caoruili/omni-bench/label_data/waiting_0812/hy.json"
    #video_dir = "/tos-bjml-llm-code-collab/liujiaheng/omni-videos-lcr/omni_videos_v1"
    video_dir = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/omni_videos_v2"


    #model_name = "/fs-computility/llm_code_collab/liujiaheng/caoruili/models/VideoLLaMA2-7B"
    model_name = "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/models/VideoLLaMA2-7B"


    model_basename = os.path.basename(model_name)
    output_file = f"./{model_basename}_{task_name}.json"

    dataloader = VideoQADaloader(data_json_file, video_dir)

    max_duration=6000

    # 禁用Torch初始化
    disable_torch_init()

    # 初始化模型、处理器和分词器
    model, processor, tokenizer = model_init(model_path=model_name)
    model = model.eval().cuda()

    # 直接在evaluate_model中处理结果写入
    evaluate_model(dataloader, model, processor, tokenizer, output_file,max_duration)

    logging.info(f"\n评测完成，结果已实时保存到 {output_file}") # print -> logging.info