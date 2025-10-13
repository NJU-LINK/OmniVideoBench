import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import torch
import requests

from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_read
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, recall_score
import numpy as np

from humanomni import model_init, mm_infer

ds_collections = {
    'emotion': {'path': '/mnt/data/qize.yqz/datasets/human/annos/1021_val_MAFW_DFEW_it_without_tag.json'}
}

from transformers import BertModel, BertTokenizer
bert_model = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

def weighted_average_recall(y_true, y_pred):
    unique_classes = np.unique(y_true)
    recalls = recall_score(y_true, y_pred, average=None, labels=unique_classes)
    print(f"{'cls':<12} | {'recall':<20}")
    for cls,recall in zip(unique_classes,recalls):
        print(f"{cls:<12} | {recall:<20.15f}")
    weights = [np.sum(np.array(y_true) == cls) for cls in unique_classes]
    total_samples = len(y_true)
    weights = np.array(weights) / total_samples
    mm=confusion_matrix(y_true, y_pred)
    print(mm)

    war = np.sum(weights * recalls)
    return war*100

def unweighted_average_recall(y_true, y_pred):
    recalls = recall_score(y_true, y_pred, average=None)
    uar = np.mean(recalls)
    return uar*100

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, ds):
        path = ds['path']
        self.datas = json.load(open(path))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        video = data['video']
        clip_meta_path = data['clip_meta_path']

        # prompt = data['conversations'][0]['value'].replace("<video>\n<audio>\n", "")
        gt =  data['conversations'][1]['value']
        source = 'mafw' if "MAFW" in video else 'dfew'
        base_prompt = "As an emotional recognition expert, in the video, when the characters display their emotions, which predominant feeling is most clearly expressed?\n"
        options_dfew = "happy ,surprise ,neutral ,angry ,disgust ,sad ,fear"
        options_mafw = "happy ,surprise ,neutral ,angry ,disgust ,sad ,fear ,contemptuous, disappointed, helpless, anxious"
        prompt = base_prompt+options_mafw if "MAFW" in video else  base_prompt+options_dfew
        return {
            'video': video,
            'prompt': prompt,
            'gt': gt,
            'clip_meta_path': clip_meta_path,
            'source': source
        }


def collate_fn(inputs, processor):
    input_texts = [_['prompt'] for _ in inputs]
    source = [_['source'] for _ in inputs]
    gt = [_['gt'] for _ in inputs]
    input_videos = [_['video'] for _ in inputs]
    input_allinone = [ _['clip_meta_path'] for _ in inputs]
   
    return input_texts, input_videos, input_allinone, gt, source


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)
    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, processor, tokenizer = model_init(args.checkpoint, device_map='cuda')




    random.seed(args.seed)
    dataset = AudioDataset(
        ds=ds_collections[args.dataset],
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
    )

    gts = []
    sources = []
    rets = []
    video_paths = []



    for _, (inputs, video_path, allinones, gt, source) in tqdm(enumerate(data_loader)):
        audio_tensor = processor["audio"](video_path[0])[0]
        video_tensor = processor["video"](video_path[0])
        # print(audio_tensor.size(), video_tensor.size())
        output = mm_infer(
            image_or_video=video_tensor,
            instruct=inputs[0],
            model=model,
            tokenizer=tokenizer,
            audio=audio_tensor,
            modal='video_audio',
            do_sample=False,
            question=inputs[0],
            bert_tokeni=bert_tokenizer
        )
        print(inputs[0], video_path[0], output, gt[0])
        gts.extend(gt)
        rets.append(output)
        sources.extend(source)
        video_paths.extend(video_path)

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_gts = [None for _ in range(world_size)]
    merged_sources = [None for _ in range(world_size)]
    merged_responses = [None for _ in range(world_size)]
    merged_video_paths = [None for _ in range(world_size)]

    torch.distributed.all_gather_object(merged_gts, gts)
    torch.distributed.all_gather_object(merged_sources, sources)
    torch.distributed.all_gather_object(merged_responses, rets)
    torch.distributed.all_gather_object(merged_video_paths, video_paths)

    merged_gts = [_ for _ in itertools.chain.from_iterable(merged_gts)]
    merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]
    merged_video_paths = [_ for _ in itertools.chain.from_iterable(merged_video_paths)]
    merged_responses = [
        _ for _ in itertools.chain.from_iterable(merged_responses)
    ]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")

        results = []
        for gt, response, source, video_path in zip(merged_gts, merged_responses, merged_sources, merged_video_paths):
            results.append({
                'gt': gt,
                'response': response,
                'source': source,
                'video_path': video_path
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{args.dataset}_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))
        results_dict = {}
        for item in tqdm(results):
            source = item["source"]
            results_dict.setdefault(source, []).append(item)

        for source in results_dict:
            refs, hyps = [], []
            results_list = results_dict[source]
            for result in results_list:
                gt = result["gt"]
                response = result["response"].lstrip()
                refs.append(gt)
                hyps.append(response)
            score = accuracy_score(refs, hyps)
            war = weighted_average_recall(refs, hyps)
            uar = unweighted_average_recall(refs, hyps)
            print(f"{source} acc: {score:.2f}%\t war: {war:.2f}% \t uar: {uar:.2f}% len:{len(hyps)}")


    torch.distributed.barrier()

"""

python -m torch.distributed.launch --use_env --master_port=29501 --nproc_per_node 8 --nnodes 1 \
    humanomni/eval/eval_mafw_dfew.py \
    --checkpoint HumanOmni_7B/ \
    --dataset  emotion
"""