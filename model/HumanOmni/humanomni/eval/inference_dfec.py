import os
import re
import math
import json
import argparse
import warnings
import traceback

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
import random
import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
withaudio= True
# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

from transformers import BertModel, BertTokenizer
bert_model = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class MVBenchDataset(Dataset):

    def __init__(self, data_list, question_list, processor, processor_audio=None):
        self.data_list = data_list
        self.processor = processor
        self.question_list = question_list
        self.processor_audio = processor_audio

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        bound = (None, None)
        video_path = self.data_list[idx]
        video_path = os.path.join('/mnt/data/jiaxing.zjx/datasets/DFEC_CVPR/test/', video_path)
        try: 
            torch_imgs = self.processor(video_path, s=bound[0], e=bound[1])
            if self.processor_audio is not None:
                audio = self.processor_audio(video_path, s=bound[0], e=bound[1])
        except Exception as e:
            backup_idx = random.randint(0, len(self.data_list)-1)
            print(f"Encounted error when reading video {video_path}, use {backup_idx}-th example instead!!!")
            return self.__getitem__(backup_idx)
        torch_imgs = self.processor(video_path, s=bound[0], e=bound[1])
        question = self.question_list[idx]
        if self.processor_audio is not None:
            audio = self.processor_audio(video_path, s=bound[0], e=bound[1])
        else:
            audio = None
        return {
            'video': torch_imgs,
            'video_path': video_path,
            'instruction': question,
            'question': question,
            'audio': audio
        }

def load_file(test_file):
    with open(test_file, 'r') as f:
        datas = json.load(f)
    video_paths = [data['video'] for data in datas]
 #   question =  "Please provide a detailed description of the facial appearance attributes and expression changes of the character in the video, including their expression state at the beginning and end of the video."
    question = "Please provide a detailed description of the facial appearance attributes and expression changes of the character in the video"
    questions = [question] * len(video_paths)
    return video_paths, questions

def build_mvbench_eval(args, processor, processor_audio=None):
    video_paths, questions = load_file(args.question_file)
    dataset = MVBenchDataset(video_paths, questions, processor, processor_audio)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return dataloader


def mvbench_dump(vid, instruct, letters, options, output):
    
    output = output.replace('answer', '')
    output = output.replace('Answer', '')
    pred_answer = re.findall(f'[\(,\ ]*[{letters[0]}-{letters[-1]}][\),\ ]*', output)
    try:
        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                # Arabic numerals -> English words
                if opt.lower() in output.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
            find_flag = True

        assert find_flag, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(vid, instruct, output)
    except:
        traceback.print_exc()
        pred_idx = 2
    
    return pred_idx


def run_inference(args):
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path)


    if withaudio:
        val_loader = build_mvbench_eval(args, processor['video'], processor['audio'])
    else:
        val_loader = build_mvbench_eval(args, processor['video'])
    results = []
    # NOTE: only support batch size 1 for now
    for i, line in enumerate(tqdm(val_loader)):
        video_tensor = line['video'][0].to(args.device)
        question = line['question'][0]
        instruct = line['instruction'][0]
        video_path = line['video_path'][0]

        if withaudio:
            audio = line['audio'][0]
        else:
            audio = None

        if withaudio:
            output = mm_infer(video_tensor, instruct, model=model, tokenizer=tokenizer, modal='video_audio', question=question,bert_tokeni=bert_tokenizer,do_sample=False, audio=audio)
        else:
            output = mm_infer(video_tensor, instruct, model=model, tokenizer=tokenizer, modal='video', question=question,bert_tokeni=bert_tokenizer,do_sample=False, audio=audio)

   
        result = {"video_path": video_path.replace('/mnt/data/jiaxing.zjx/datasets/DFEC_CVPR/test/', ''), "instruction": question, "output": output}
        results.append(result)

    with open(args.answer_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--question-file', required=True)
    parser.add_argument('--answer-file', required=True)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)

    args = parser.parse_args()
    run_inference(args)
