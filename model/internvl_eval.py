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
from decord import VideoReader, cpu
import cv2
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import random

import sys
sys.path.append("/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench")
from dataloader import VideoQADaloader

import torch.distributed as dist

# dist.init_process_group(backend="nccl")
# torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

def load_video_opencv(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """使用OpenCV加载视频的备选方案"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"OpenCV无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f'视频总帧数: {total_frames}, FPS: {fps}')
        
        # 计算帧索引
        frame_indices = get_index(bound, fps, total_frames - 1, first_idx=0, num_segments=num_segments)
        
        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        
        for frame_index in frame_indices:
            # 设置帧位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if not ret:
                print(f"无法读取帧 {frame_index}, 跳过")
                continue
                
            # 转换颜色空间 BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # 动态预处理
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        
        cap.release()
        
        if not pixel_values_list:
            raise Exception("未能成功读取任何视频帧")
            
        pixel_values = torch.cat(pixel_values_list)
        
        print(f'提取了 {len(frame_indices)} 帧，总图片块数: {pixel_values.shape[0]}')
        return pixel_values, num_patches_list
        
    except Exception as e:
        print(f"OpenCV视频加载失败: {e}")
        raise e

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """加载视频并提取帧，支持DECORD和OpenCV备选方案"""
    try:
        # 首先尝试使用DECORD
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        print(f'视频总帧数: {max_frame + 1}, FPS: {fps}')

        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        
        print(f'提取了 {len(frame_indices)} 帧，总图片块数: {pixel_values.shape[0]}')
        return pixel_values, num_patches_list
        
    except Exception as decord_error:
        print(f"DECORD加载失败: {decord_error}")
        
        # 检查是否是特定的DECORD错误
        error_msg = str(decord_error)
        if ("cannot find video stream" in error_msg or 
            "DECORDError" in error_msg or 
            "Invalid NAL unit size" in error_msg or
            "Error splitting the input into NAL units" in error_msg or
            "Error sending packet" in error_msg or
            "avcodec_send_packet" in error_msg or
            "DECORD加载失败" in error_msg):
            
            print("检测到DECORD兼容性问题，尝试使用OpenCV备选方案...")
            try:
                return load_video_opencv(video_path, bound, input_size, max_num, num_segments)
            except Exception as opencv_error:
                print(f"OpenCV备选方案也失败: {opencv_error}")
                # 抛出原始的DECORD错误，保持错误信息的一致性
                raise decord_error
        else:
            # 对于其他类型的错误，直接抛出
            raise decord_error

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

def evaluate_model(dataloader, model, tokenizer, output_file, max_duration, num_segments=8, max_num=1):
    """使用InternVL模型进行评估"""
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
    generation_config = dict(max_new_tokens=1024, do_sample=True)

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
            # 加载视频并提取帧
            pixel_values, num_patches_list = load_video(
                video_path, 
                num_segments=num_segments, 
                max_num=max_num
            )
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            
            # 验证内容完整性
            if pixel_values.size(0) == 0:
                print(f"视频内容提取失败，跳过: {video_name}")
                result["model_answer"] = "Error: 视频内容提取失败"
                write_result_to_file(result, output_file)
                PROBLEMATIC_VIDEOS.add(video_path)
                error_count += 1
                continue
            
            print(f"视频处理完成: {pixel_values.shape[0]} 个图片块")

            question = item.get("question")
            correct_answer = item.get("answer")  # 修复：使用 "answer" 而不是 "correct_option"
            #TODO: 删除
            print(f"question: {question}")
            print(f"correct_answer: {correct_answer}")

            # 构建视频前缀
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            
            question_prompt = (
                f"{video_prefix}"
                "You are given a video. Based on the content of the video, answer the following question:\n\n"
                f"Question:\n{question}\n\n"
                f"Options:\n{options_text}\n\n"
                "Answer with the option's letter directly(e.g., A, B, C, or D). "
                "If your access to the video content is limited, at least one option that is more likely than the others must be chosen. "
                "Mustn't give any other reason for can not choose!"
            )

            # 模型推理，包含错误捕获
            try:
                response = model.chat(
                    tokenizer, 
                    pixel_values, 
                    question_prompt, 
                    generation_config,
                    num_patches_list=num_patches_list
                )
                
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
    parser.add_argument("--data_json_file", type=str, default="/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/final_data/out.json")
    parser.add_argument("--video_dir", type=str, default="/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/omni_videos_v1")
    parser.add_argument("--output_file", type=str, default="/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/eval_results/internvl_38b_128frames_out.json")
    parser.add_argument("--model_path", type=str, default="/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/models/InternVL3_5-38B")
    parser.add_argument("--max_duration", type=int, default=6000)
    parser.add_argument("--num_segments", type=int, default=128)
    parser.add_argument("--max_num", type=int, default=1)
    args = parser.parse_args()

    data_json_file = args.data_json_file
    video_dir = args.video_dir
    output_file = args.output_file
    model_path = args.model_path
    max_duration = args.max_duration
    num_segments = args.num_segments
    max_num = args.max_num
    
    dataloader = VideoQADaloader(data_json_file, video_dir)
    
    # 加载InternVL模型
    print("正在加载 InternVL 模型...")
    
    # 在分布式环境中避免使用device_map="auto"，改为手动指定设备
    # if dist.is_initialized():
    #     # 分布式环境：使用当前进程的GPU
    #     device = f"cuda:{local_rank}"
    #     model = AutoModel.from_pretrained(
    #         model_path,
    #         torch_dtype=torch.bfloat16,
    #         load_in_8bit=False,
    #         low_cpu_mem_usage=True,
    #         use_flash_attn=True,
    #         trust_remote_code=True,
    #         device_map={"": device}  # 手动指定设备映射
    #     ).eval()
    # else:
    #     # 非分布式环境：可以使用auto或手动指定
    #     try:
    def init_distributed():
        if not dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 由accelerate自动设置为0/1（2卡）
            torch.cuda.set_device(local_rank)  # 绑定当前进程到对应GPU
            # 初始化进程组（backend用nccl，适合GPU间通信）
            dist.init_process_group(backend="nccl")
        return int(os.environ.get("LOCAL_RANK", 0))

    # 2. 先初始化分布式，再加载模型
    local_rank = init_distributed()
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
        #device_map={"": local_rank}
    ).eval()
    print(model.hf_device_map)
        # except Exception as e:
        #     print(f"使用device_map='auto'失败: {e}")
        #     print("尝试手动指定设备...")
        #     # 备选方案：手动指定到第一个GPU
        #     model = AutoModel.from_pretrained(
        #         model_path,
        #         torch_dtype=torch.bfloat16,
        #         load_in_8bit=False,
        #         low_cpu_mem_usage=True,
        #         use_flash_attn=True,
        #         trust_remote_code=True,
        #         device_map={"": "cuda:0"}
        #     ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        use_fast=False
    )
    
    print("模型加载完成，开始评估...")
    
    # 直接在evaluate_model中处理结果写入
    eval_results = evaluate_model(
        dataloader, 
        model, 
        tokenizer, 
        output_file, 
        max_duration,
        num_segments=num_segments,
        max_num=max_num
    )

    print(f"\n评测完成，结果已实时保存到 {output_file}")
    print(f"最终统计结果:")
    print(f"  - 总处理数量: {eval_results['total_processed']}")
    print(f"  - 正确回答数: {eval_results['correct_answers']}")
    print(f"  - 最终准确率: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
