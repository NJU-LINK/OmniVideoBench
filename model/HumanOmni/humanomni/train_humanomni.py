# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


import ipdb
import re
import os
import copy
import json
import random
import pathlib
import traceback
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import h5py
from transformers import BertModel, BertTokenizer
# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
from torch.utils.data import Dataset

from PIL import Image
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
from transformers import StoppingCriteria
import transformers
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from filelock import FileLock
import sys
sys.path.append('./')
from humanomni.model import *
from humanomni.constants import  NUM_FRAMES, IGNORE_INDEX, MODAL_INDEX_MAP, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN
from humanomni.mm_utils import tokenizer_multimodal_token, process_image, process_video, read_video_patch, process_audio, frame_sample
from humanomni.humanomni_trainer import (HumanOmniTrainer, 
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    find_all_linear_names, safe_save_model_for_hf_trainer
)

import torchaudio
import math 
import yaml



def sample_frames(largest_body_tracklets, num_frames):
    if not largest_body_tracklets:
        return {}

    frame_indices = sorted(largest_body_tracklets.keys())
    total_frames = len(frame_indices)
    
    if total_frames <= num_frames:
        sampled_frame_indices = frame_indices
    else:
        step = total_frames // num_frames
        sampled_frame_indices = frame_indices[::step][:num_frames]
    
   # sampled_tracklets = {frame_idx: largest_body_tracklets[frame_idx] for frame_idx in sampled_frame_indices}
    
    return sampled_frame_indices

# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486   
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None
bert_model = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model)


# 设置 NCCL 的阻塞等待模式和超时（单位为秒）
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"  # 可选，调试时使用
# 设置超时时间为3分钟（180秒）
os.environ["NCCL_TIMEOUT"] = "180"



def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelArguments:
    # LLM Arguments
    model_type: Optional[str] = field(default="HumanOmni_qwen2", metadata={"help": "Model type selected in the list: " + ", ".join(VLLMs.keys())})
    model_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )

    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='linear')
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # Vision tower Arguments
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    bert_type: Optional[str] = field(default="qformer_pretrain")
    num_query: Optional[int] = field(default=32)
    pretrain_qformer: Optional[str] = field(default=None)
    compress_type: Optional[str] = field(default=None)
    use_vid:Optional[bool] = field(default=False)

    audio_tower: Optional[str] = field(default=None)
    pretrain_audio_mlp_adapter: Optional[str] = field(default=None)
    mm_audio_projector_type: Optional[str] = field(default='mlp2x_gelu')

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_folder: Optional[str] = field(default=None)
    # Loading Arguments
    is_multimodal: bool = False
    lazy_preprocess: bool = False
    num_frames: Optional[int] = field(default=None)
    # Preprocess Arguments
    mm_use_x_start_end: bool = False
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    load_version: str=''


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def add_audio_token_to_sources(sources):
    for item in sources:
        for conv in item['conversations']:
            if conv['from'] == 'human':
                value = conv['value']
                # 检查当前的标记状态
                has_video = '<video>' in value
                has_audio = '<audio>' in value
                
                if has_video and has_audio:
                    # 已经同时包含两个标记，不需要修改
                    continue
                elif has_video and not has_audio:
                    # 只有 video，添加 audio
                    conv['value'] = value.replace('<video>\n', '<video>\n<audio>\n')
                elif has_audio and not has_video:
                    # 只有 audio，添加 video
                    conv['value'] = value.replace('<audio>\n', '<video>\n<audio>\n')
                # 如果两个都没有的情况，保持原样不变
    return sources

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
  #  import ipdb;ipdb.set_trace()
    roles = {"human": "user", "gpt": "assistant"}
    conversations = []
    input_ids = []
    targets = []
    for source in sources:
        assert len(source) == 2
        assert modal_token in source[0]['value']
        message = [
            {'role': 'user', 'content': modal_token},
            {'role': 'assistant', 'content': source[1]['value']}
        ]
        conversation = " ".join([sentence['value'] for sentence in source])
        input_id = tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt')
        target = copy.deepcopy(input_id)
        target[input_id == MODAL_INDEX_MAP[modal_token]] = IGNORE_INDEX
        input_ids.append(input_id)
        targets.append(target)
    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    roles_ = ["user", "assistant"]

    conversations = []
    input_ids = []
    targets = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != "user":
            # Skip the first one if it is not from human
            source = source[1:]

        message = [{'role': roles[sentence['from']], 'content': sentence['value']} for sentence in source]
        conversation = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        input_ids.append(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))
        targets.append(copy.deepcopy(input_ids[-1]))

        assert len(source) % 2 == 0, f"Invalid conversation length {len(source)}."

        cur = 0
        message = []
        for idx, sentence in enumerate(source):
            if idx % 2 == 1:
                tmp_message = [
                    {'role': roles[source[idx-1]['from']], 'content': source[idx-1]['value']}, 
                    {'role': roles[sentence['from']], 'content': sentence['value']}
                ]

                instruction = tokenizer.apply_chat_template(message + tmp_message[:1], tokenize=False, add_generation_prompt=True)
                conversation = tokenizer.apply_chat_template(message + tmp_message, tokenize=False, add_generation_prompt=False)

                instruction_len = len(tokenizer_multimodal_token(instruction, tokenizer, modal_token, return_tensors='pt'))
                conversation_len = len(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))

                targets[-1][cur:instruction_len] = IGNORE_INDEX

                cur = conversation_len
                message += tmp_message
    
    guided_prompt = []
    refine_prompt=False
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != "user":
            # Skip the first one if it is not from human
            source = source[1:]


        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == roles_[j % 2], f"{i}"
            
            # add guided prompt
            if role=="user":
                guided_sent = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '').replace(DEFAULT_VIDEO_TOKEN, '')
                
                if refine_prompt:
                    # only keep the useful part of the prompt
                    if '\n' in guided_sent:
                        for _sent in guided_sent.split('\n'):
                            if '?' in _sent:
                                guided_sent = _sent
                                break
                guided_prompt.append(guided_sent) 

    return dict(input_ids=input_ids, labels=targets, prompts=guided_prompt)


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            for DEFAULT_TOKEN in MODAL_INDEX_MAP.keys():
                # TODO: hacky here
                replace_token = DEFAULT_TOKEN
                X = DEFAULT_TOKEN[1:-1]
                if DEFAULT_TOKEN in sentence['value']:
                    # put the tag ahead of the sentence
                    sentence['value'] = sentence['value'].replace(f"{DEFAULT_TOKEN}", '').replace(DEFAULT_TOKEN, '').strip()
                    sentence['value'] = replace_token + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                if data_args.mm_use_x_start_end:
                    new_replace_token = DEFAULT_X_START_TOKEN[X.upper()] + replace_token + DEFAULT_X_END_TOKEN[X.upper()]
                    sentence["value"] = sentence["value"].replace(replace_token, new_replace_token)
    return sources

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = [ ]
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None
                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")
                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")
                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)
                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)

        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            if data_path.endswith(".jsonl"):
                cur_data_dict = []
                with open(data_path, "r") as json_file:
                    for line in json_file:
                        cur_data_dict.append(json.loads(line.strip()))
            elif data_path.endswith(".json"):
                with open(data_path, "r") as json_file:
                    cur_data_dict = json.load(json_file)
            rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
            self.list_data_dict.extend(cur_data_dict)
            
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 576 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
      #  ipdb.set_trace()
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME


        image_processor = self.data_args.image_processor
        video_processor = self.data_args.video_processor

        audio_processor = getattr(self.data_args, "audio_processor", None)
        if audio_processor:
            audio = torch.zeros(1, audio_processor.feature_size, audio_processor.nb_max_frames)


        num_frames = NUM_FRAMES if self.data_args.num_frames is None else self.data_args.num_frames


        if audio_processor is not None:
            sources = add_audio_token_to_sources(sources)
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_file_origin = image_file
            image_folder = self.data_args.data_folder
            image_file = os.path.join(image_folder, image_file)
            try:
                image = process_image(image_file, image_processor, aspect_ratio=self.data_args.image_aspect_ratio)
            except:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                print(f"Encounted error when reading image {image_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)

            modal_token = "<image>"
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        elif 'video' in sources[0]:
            video_file = self.list_data_dict[i]['video']
            video_file_origin = video_file
            video_folder = self.data_args.data_folder
            video_file = os.path.join(video_folder, video_file)    
            try:
                video = process_video(video_file, video_processor, aspect_ratio=self.data_args.image_aspect_ratio, num_frames=num_frames)
            except Exception as e:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                print(f"Encounted error when reading video {video_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)
            
            if audio_processor is not None:
                try: 
                    audio, audio_sample_rate = process_audio(video_file)
                    audio = audio_processor(audio, sampling_rate=audio_sample_rate, return_tensors='pt')['input_features']
                except Exception as e:
                    traceback.print_exc()
                    audio=None
                    print(f"Encounted error when reading video {video_file}, not audio found!!!")

            modal_token = "<video>"
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        elif 'audio' in sources[0]:
            try:
                modal_token = '<audio>'
                audio, sampling_rate =  torchaudio.load(self.list_data_dict[i]['audio'], normalize=True, channels_first=True) # TODO: Check norm
                audio = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=audio_processor.sampling_rate)(audio)
                if len(audio.shape)>1 and sampling_rate!=-1:
                    audio = audio.mean(dim=0)
                audio_processor = getattr(self.data_args, "audio_processor", None)
                audio = audio_processor(audio, sampling_rate=audio_processor.sampling_rate, return_tensors='pt')['input_features']
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            except Exception as e:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                return self.__getitem__(backup_idx)

        else:
            modal_token = None
            sources = copy.deepcopy([e["conversations"] for e in sources])

        if self.data_args.is_pretraining:
            data_dict = preprocess(sources, self.tokenizer, modal_token=modal_token)
        else:
            data_dict = preprocess(sources, self.tokenizer, modal_token=modal_token)

        if isinstance(i, int):
            if "prompts" in data_dict.keys():
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0], prompts=data_dict["prompts"])
            else:
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif 'video' in self.list_data_dict[i]:
            data_dict['video'] = video

        elif self.data_args.is_multimodal:
            data_dict['image'] = torch.zeros(3, self.data_args.image_size, self.data_args.image_size)

        if audio_processor:
            data_dict['audio'] = audio

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # work for 'images' argument in `prepare_inputs_labels_for_multimodal` of LlavaMetaForCausalLM in llava_arch.py
        batch['images'] = []

        for instance in instances:
            for modal_token in ["<image>", "<video>"]:
                modal_token = modal_token.lower()
                # MODAL_TOKEN shape like: <image>, <video>, ...
                modal_name = re.findall(f'[<](.*)[>]', modal_token)
                assert len(modal_name) == 1
                modal_name = modal_name[0]
                if modal_name in instance:
                    batch['images'].append((instance[modal_name], modal_name))

        batch_size = len(batch['images'])


        if 'audio' in instances[0]:
            audios = [instance['audio'] for instance in instances]
            if all(x is not None and audios[0] is not None and x.shape == audios[0].shape for x in audios):
                batch['audios'] = torch.cat(audios, dim=0)
            else:
                batch['audios'] = audios

        if 'prompts' in instances[0]:
            batch['prompts'] = [instance['prompts'] for instance in instances]
        prompts_bert = []
        for prompt in batch['prompts']:
            prompts_bert.append(prompt[0])
        inputs_bert = bert_tokenizer(prompts_bert, return_tensors='pt', padding=True, truncation=True,add_special_tokens=True)
        batch['prompts'] = inputs_bert

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def check_parameters(model):
    frozen_params = []
    trainable_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print("Frozen Parameters:")
    for name in frozen_params:
        print(name)
    
    print("\nTrainable Parameters:")
    for name in trainable_params:
        print(name)

def train(attn_implementation=None):
    global local_rank
    set_seed(42)
   # import ipdb; ipdb.set_trace()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # device_map={"": training_args.device},
            # BUG: High version transformers report error: 
            # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))

    config = VLLMConfigs[model_args.model_type].from_pretrained(model_args.model_path, trust_remote_code=True)
    config._attn_implementation = attn_implementation

    if model_args.vision_tower is not None:
        model = VLLMs[model_args.model_type].from_pretrained(
            model_args.model_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
        if 'mixtral' in model_args.model_type:
            import deepspeed
            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if "_llama3.1" in model_args.model_type:
        # tokenizer.eos_token_id = 128001
        tokenizer.pad_token = "<|end_of_text|>"
        tokenizer.pad_token_id = 128001


    if model_args.vision_tower is not None or model_args.audio_tower is not None: 
        if model_args.vision_tower is not None:
            # initialize vision encoder + multi-modal projector
            model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
            if model_args.use_vid:
                model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
                model.get_model().initialize_attention_modules(model_args)
            vision_tower = model.get_vision_tower()
            vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            data_args.image_size = vision_tower.image_size
            data_args.image_processor = vision_tower.image_processor
            data_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor
            data_args.is_multimodal = True
            if data_args.image_grid_pinpoints is not None:
                if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                    try:
                        patch_size = data_args.image_processor.size[0]
                    except Exception as e:
                        patch_size = data_args.image_processor.size["height"]
                    assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
                    # Use regex to extract the range from the input string
                    matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                    range_start = tuple(map(int, matches[0]))
                    range_end = tuple(map(int, matches[-1]))
                    # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                    grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
                    # Multiply all elements by patch_size
                    data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
                elif isinstance(data_args.image_grid_pinpoints, str):
                    data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)
        
        if model_args.audio_tower is not None:
            audio_tower = model.get_audio_tower()
            if audio_tower is None:
                rank0_print("initialize audio modules")
                model.get_model().initialize_audio_modules(
                    model_args=model_args,
                    fsdp=training_args.fsdp
                )
                audio_tower = model.get_audio_tower()
            if not audio_tower.is_loaded:
                rank0_print("load audio modules")
                audio_tower.load_model()
            audio_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            data_args.audio_processor = audio_tower.audio_processor
     
            data_args.is_multimodal = True


        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.mm_use_x_start_end = model_args.mm_use_x_start_end = data_args.mm_use_x_start_end
        training_args.use_x_start_end = model_args.mm_use_x_start_end


        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

        data_args.is_pretraining = True
        ### Deciding train which part of the model
        if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            if model_args.tune_mm_mlp_adapter:
                model.requires_grad_(False)

                tune_modules = ['mm_projector', 'pos_emb', 'audio_projector', 'vision_sampler']
                for name, param in model.named_parameters():
                    if any(listed_name in name for listed_name in tune_modules):
                        rank0_print('tuning {}'.format(name))
                        param.requires_grad = True
            else:
                data_args.is_pretraining = False
            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                tune_modules = ['mm_projector', 'pos_emb', 'audio_projector', 'vision_sampler']
                for name, param in model.named_parameters():
                    if any(listed_name in name for listed_name in tune_modules):
                        rank0_print('freezing {}'.format(name))
                        param.requires_grad = False
        else:
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
            # Set the entire model to not require gradients by default
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            if model_args.audio_tower is not None:
                model.get_model().audio_projector.requires_grad_(False)
            # Parse the mm_tunable_parts to decide which parts to unfreeze
            tunable_parts = model_args.mm_tunable_parts.split(",")
            tunable_parts = [part.strip() for part in tunable_parts]
            if "mm_mlp_adapter" in tunable_parts:
                tune_modules = ['image_newline', "mm_projector"]
                for name, param in model.named_parameters():
                    if any(listed_name in name for listed_name in tune_modules):
                        # rank0_print('tuning {}'.format(name))
                        param.requires_grad = True
            if "audio_projector" in tunable_parts:
                for p in model.get_model().audio_projector.parameters():
                    p.requires_grad = True
            if "mm_vision_resampler" in tunable_parts:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                data_args.is_pretraining = False
                for name, param in model.named_parameters():
                    if ("vision_tower" not in name and "image_newline" not in name and "aux_encoder" not in name and
                    "mm_projector" not in name and "vision_resampler" not in name and 
                    "audio_tower" not in name and "audio_projector" not in name):
                        param.requires_grad_(True)

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.num_frames = NUM_FRAMES if data_args.num_frames is None else data_args.num_frames
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    for p in model.get_model().bert_model.parameters():
        p.requires_grad = False
    for p in model.get_model().bert_gate.parameters():
        p.requires_grad = True
        
    check_parameters(model)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)


    # select a Trainer

    trainer = HumanOmniTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()