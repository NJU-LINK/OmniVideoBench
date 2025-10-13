import os
import json
import math
import tempfile
from typing import List, Dict, Any
import gc
import psutil
import re

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, GenerationConfig
import random

import sys
sys.path.append("/root/siton-tmp/code/omni-bench")
from dataloader import VideoQADaloader

# 添加Ming模型路径
sys.path.append("/root/siton-tmp/code/omni-bench/model/Ming")
from modeling_bailingmm import BailingMMNativeForConditionalGeneration

import torch.distributed as dist

# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
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

def extract_answer_letter(response):
    """
    使用正则表达式从模型回答中提取答案字母 (A, B, C, D)
    """
    if not response:
        return ""
    
    # 清理回答文本
    response = response.strip()
    
    # 尝试多种正则模式来提取答案
    patterns = [
        r'^([ABCD])\.?\s*',  # Starts with A. or A
        r'([ABCD])\.?\s*$',  # Ends with A. or A
        r'answer\s*(is|:)?\s*([ABCD])',  # answer is A or answer: A
        r'choose\s*([ABCD])',  # choose A
        r'option\s*([ABCD])',  # option A
        r'([ABCD])\s*option',  # A option
        r'\b([ABCD])\b',  # single letter
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 如果没有找到，返回原始回答的第一个字符（如果是A-D）
    first_char = response[0].upper() if response else ""
    if first_char in ['A', 'B', 'C', 'D']:
        return first_char
    
    return ""

def generate_ming_response(messages, processor, model):
    """
    使用Ming-omni模型生成响应
    """
    try:
        # 1. Format inputs using chat template
        text = processor.apply_chat_template(messages, add_generation_prompt=True)

        # 2. Extract vision/audio data
        image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

        # 3. Prepare tensor inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        for k in inputs.keys():
            if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        # 4. Configure generation
        generation_config = GenerationConfig.from_dict({'no_repeat_ngram_size': 10})
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            use_cache=True,
            eos_token_id=processor.gen_terminator,
            generation_config=generation_config,
        )
        generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

        # 5. Decode output
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
        
    except Exception as e:
        print(f"Ming模型推理错误: {e}")
        raise e

def filter_qa_pairs_by_duration(qa_pairs, max_duration):
    """Filter QA pairs based on video duration."""
    filtered_qa_pairs = []
    for qa_pair in qa_pairs:
        duration = qa_pair.get("duration")
        if duration is not None and duration < max_duration:
            filtered_qa_pairs.append(qa_pair)
        elif duration is None:
            print(f"Warning: No duration data for video {qa_pair.get('video_path', 'Unknown')}, skipping.")
    
    print(f"原始数据集: {len(qa_pairs)} 个QA对")
    print(f"过滤后数据集 < {max_duration} 秒: {len(filtered_qa_pairs)} 个QA对")
    return filtered_qa_pairs

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

def evaluate_model(dataloader, model, processor, output_file, max_duration, max_frames=64):
    """使用Ming-omni模型进行评估"""
    # 使用filter_qa_pairs_by_duration过滤QA对
    qa_pairs = filter_qa_pairs_by_duration(dataloader.get_all_qa_pairs(), max_duration)
    
    # 内存监控
    print(f"开始评估，初始内存使用: {psutil.virtual_memory().percent}%")
    print(f"过滤后的QA对数量: {len(qa_pairs)}")
    torch.cuda.empty_cache()
    gc.collect()

    processed_count = 0
    error_count = 0
    correct_count = 0

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
            "options": item.get("options", []),
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
            question = item.get("question")
            correct_answer = item.get("answer")
            
            print(f"question: {question}")
            print(f"correct_answer: {correct_answer}")

            # 构建Ming-omni的消息格式
            question_prompt = (
                "You are given a video. Based on the content of the video, answer the following question:\n\n"
                f"Question:\n{question}\n\n"
                f"Options:\n{options_text}\n\n"
                "Answer with the option's letter directly(e.g., A, B, C, or D). "
                "If your access to the video content is limited, at least one option that is more likely than the others must be chosen. "
                "Mustn't give any other reason for can not choose!"
            )

            # 构建Ming消息格式
            messages = [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "video", "video": video_path, "max_frames": max_frames, "sample": "uniform"},
                        {"type": "text", "text": question_prompt},
                    ],
                },
            ]

            # 模型推理，包含错误捕获
            try:
                response = generate_ming_response(messages, processor, model)
                
                print(f"response: {response}")

                # 使用正则表达式提取答案字母
                extracted_answer = extract_answer_letter(response)
                model_answer = extracted_answer if extracted_answer else response.strip()
                
                print(f"extracted_answer: {extracted_answer}")
                
                is_correct = (extracted_answer.upper() == correct_answer.strip().upper()) if extracted_answer else False

                result["model_answer"] = model_answer
                result["is_correct"] = is_correct
                processed_count += 1
                
                # 统计正确答案数量
                if is_correct:
                    correct_count += 1
                
            except torch.cuda.OutOfMemoryError as oom_error:
                error_msg = f"Error: OutOfMemoryError - CUDA out of memory. {str(oom_error)}"
                print(f"CUDA内存不足: {oom_error}")
                result["model_answer"] = error_msg
                result["is_correct"] = False
                error_count += 1
                
                # 清理内存并继续
                torch.cuda.empty_cache()
                gc.collect()
                
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
    
    # 计算准确率
    accuracy = correct_count / processed_count if processed_count > 0 else 0.0
    
    print(f"\n评估完成!")
    print(f"成功处理: {processed_count} 个视频")
    print(f"正确回答: {correct_count} 个")
    print(f"错误数量: {error_count} 个")
    print(f"问题视频数量: {len(PROBLEMATIC_VIDEOS)}")
    print(f"准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        "total_processed": processed_count,
        "correct_answers": correct_count,
        "errors": error_count,
        "accuracy": accuracy,
        "problematic_videos": len(PROBLEMATIC_VIDEOS)
    }

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json_file", type=str, default="/root/siton-tmp/code/omni-bench/final_data/out.json")
    parser.add_argument("--video_dir", type=str, default="/root/siton-tmp/omni-videos/omni_videos_v3")
    parser.add_argument("--output_file", type=str, default="/root/siton-tmp/code/omni-bench/eval_results/ming_omni_out.json")
    parser.add_argument("--model_path", type=str, default="/root/siton-tmp/code/models/Ming-Lite-Omni-1.5")
    parser.add_argument("--max_duration", type=int, default=6000)
    parser.add_argument("--max_frames", type=int, default=128) # 参考原论文的设置
    args = parser.parse_args()

    data_json_file = args.data_json_file
    video_dir = args.video_dir
    output_file = args.output_file
    model_path = args.model_path
    max_duration = args.max_duration
    max_frames = args.max_frames
    
    dataloader = VideoQADaloader(data_json_file, video_dir)
    
    # 加载Ming-omni模型
    print("正在加载 Ming-omni 模型...")
    
    # def init_distributed():
    #     if not dist.is_initialized():
    #         local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #         torch.cuda.set_device(local_rank)
    #         dist.init_process_group(backend="nccl")
    #     return int(os.environ.get("LOCAL_RANK", 0))

    # # 初始化分布式，再加载模型
    # local_rank = init_distributed()
    
    # 加载Ming-omni模型
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        load_image_gen=False,  # 不需要图像生成功能
        low_cpu_mem_usage=True,
        device_map="auto"
    ).eval()
    
    # print(f"模型设备映射: {model.hf_device_map}")
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    print("模型加载完成，开始评估...")
    
    # 直接在evaluate_model中处理结果写入
    eval_results = evaluate_model(
        dataloader, 
        model, 
        processor, 
        output_file, 
        max_duration,
        max_frames=max_frames
    )

    print(f"\n评测完成，结果已实时保存到 {output_file}")
    print(f"最终统计结果:")
    print(f"  - 总处理数量: {eval_results['total_processed']}")
    print(f"  - 正确回答数: {eval_results['correct_answers']}")
    print(f"  - 最终准确率: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
