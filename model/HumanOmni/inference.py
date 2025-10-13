import os
import argparse
from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init
from transformers import BertTokenizer

# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser(description="HumanOmni Inference Script")
    parser.add_argument('--modal', type=str, default='video_audio', help='Modal type (video or video_audio)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file')
    parser.add_argument('--instruct', type=str, required=True, help='Instruction for the model')

    args = parser.parse_args()

    # 初始化BERT分词器
    bert_model = "bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

    # 禁用Torch初始化
    disable_torch_init()

    # 初始化模型、处理器和分词器
    model, processor, tokenizer = model_init(args.model_path)

    # 处理视频输入
    video_tensor = processor['video'](args.video_path)
    
    # 根据modal类型决定是否处理音频
    if args.modal == 'video_audio' or args.modal == 'audio':
        audio = processor['audio'](args.video_path)[0]
    else:
        audio = None

    # 执行推理
    output = mm_infer(video_tensor, args.instruct, model=model, tokenizer=tokenizer, modal=args.modal, question=args.instruct, bert_tokeni=bert_tokenizer, do_sample=False, audio=audio)
    print(output)

if __name__ == "__main__":
    main()
