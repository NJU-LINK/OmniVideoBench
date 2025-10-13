# HumanOmni: A Large Vision-Speech Language Model for Human-Centric Video Understanding

[![ModelScope](https://img.shields.io/badge/ModelScope-HumanOmni-blue)](https://modelscope.cn/models/iic/HumanOmni-7B)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-HumanOmni-yellow)](https://huggingface.co/StarJiaxing/HumanOmni-7B)
[![arXiv](https://img.shields.io/badge/arXiv-2501.15111-red)](https://arxiv.org/abs/2501.15111)

<div align="center">
  <img src="figures/arch.png" width="800"/>
</div>

## [![News](https://img.shields.io/badge/News-Update-blue)](#news)News 
1) Building upon HumanOmni, we are the **first to combine RLVR (Reinforcement Learning for Vision and Reasoning) with an Omni model**, introducing [R1-Omni](https://github.com/HumanMLLM/R1-Omni), a reasoning-based large multimodal model.


## 📖 Introduction
**HumanOmni** is the industry’s first human-centric Omni-multimodal large language model for comprehensive understanding in human-centric scenes.
1) **2.4M human-centric video clips with over 14M double-check instructions**: We have constructed a dataset containing over 2.4M human-centric video clips, providing rich and detailed information about individuals. We provide over 14M instruction data for visual pretraining.
2) **50K video clips with more than 100K manually annotated instrcutions**: We have manually annotated 50K video clips with more than 100K instructions related to emotion recognition, facial description, and speaker-specific speech recognition for visual fine-tuning and cross-modal interaction integration.
3) **Three human-specific branch**: We use three branches to handle face-related, body-related, and interaction-related scenes separately in HumanOmni. HumanOmni dynamically adjusts its fusion weights based on input instructions, ensuring accurate responses across various scenes.
4) **Audio-visual synergy**: HumanOmni can simultaneously understand vision and speech, allowing for a more comprehensive understanding of complex scenes.



## 📦 Model Download
<div align="center">

| **Model**              | **Stage**                          | **#Params** | **HuggingFace**                                                                 | **ModelScope**                                                          |
|------------------------|------------------------------------|-------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `HumanOmni-Video`      | Visual Capability Construction     | 7B          | [![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://hf.co/StarJiaxing/HumanOmni-7B-Video) | [![MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/iic/HumanOmni-7B-Video) |
| `HumanOmni-Audio`      | Auditory Capability Development    | 7B          | [![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://hf.co/StarJiaxing/HumanOmni-7B-Audio)  | [![MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/iic/HumanOmni-7B-Audio)  |
| `HumanOmni-Omni`       | Cross-Modal Interaction Integration | 7B          | [![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://hf.co/StarJiaxing/HumanOmni-7B)         | [![MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/iic/HumanOmni-7B)         |

</div>

Our training pipeline consists of three progressive stages to establish multimodal understanding capabilities:

📹 Visual Capability Construction

- Model: HumanOmni-Video
- Objective: Learn spatio-temporal feature representations to analyze human actions and scene dynamics in videos.
  
🎧 Auditory Capability Development

- Model: HumanOmni-Audio
- Objective: Develop robust speech comprehension and audio interpretation through large-scale acoustic modeling.

🌐 Cross-Modal Interaction Integration

- Model: HumanOmni-Omni (also referred to as HumanOmni)
- Objective: Enable synergistic vision-audio reasoning by fine-tuning parameters from both HumanOmni-Video and HumanOmni-Audio.


## 🏆 Performance

- Emotion Understanding:

| Method                          | Modalities | DFEW (UAR) | DFEW (WAR) | MAFW (UAR) | MAFW (WAR) |
|----------------------------------|------------|-------------|-------------|-------------|-------------|
| **Specialized models for emotion-related tasks** |            |             |             |             |             |
| Wav2Vec2.0                   | A          | 36.15       | 43.05       | 21.59       | 29.69       |
| HuBERT                       | A          | 35.98       | 43.24       | 25.00       | 32.60       |
| DFER-CLIP                     | V          | 59.61       | 71.25       | 38.89       | 52.55       |
| MAE-DFER                      | V          | 63.41       | 74.43       | 41.62       | 54.31       |
| HiCMAE                        | AV         | 63.76       | 75.01       | 42.65       | 56.17       |
| Emotion-LLaMA                 | AV         | 64.21       | 77.06       | -           | -           |
| MMA-DFER                         | AV         | 66.85       | 77.43       | 44.25       | 58.45       |
| **Other models**                 |            |             |             |             |             |
| Qwen2-VL-7B                  | V          | 43.08       | 52.83       | 31.67       | 45.89       |
| Qwen2-VL-72B                  | V          | 39.24       | 45.12       | 42.61       | 46.07       |
| VITA                          | AV         | 21.36       | 32.07       | 14.05       | 33.38       |
| InternLM-XComposer-2.5-OL     | AV         | 44.23       | 51.29       | 33.78       | 46.81       |
| GPT4-O                       | AV         | 50.57       | 57.19       | 38.29       | 48.82       |
| **HumanOmni**                    | AV         | **74.86**   | **82.46**   | **52.94**   | **68.40**   |

- [Dynamic Facial Expression Caption](https://modelscope.cn/datasets/iic/DFEC):

| Method                          | Correctness | Detail | Context | Temporal | CIDEr  | Rouge-L | AutoDQ  |
|----------------------------------|-------------|--------|---------|----------|--------|---------|---------|
| **Vision large language model** |             |        |         |          |        |         |         |
| VideoLLaMA                    | 3.60        | 3.67   | 3.84    | 3.50     | 0.189  | 0.196   | 0.303   |
| VideoChat                     | 3.47        | 3.52   | 3.92    | 3.38     | 0.251  | 0.192   | 0.344   |
| VideoChat2                    | 3.70        | 3.56   | 4.16    | 3.52     | 0.202  | 0.229   | 0.311   |
| Chat-UniVI                    | 3.64        | 3.63   | 4.21    | 3.61     | 0.189  | 0.231   | 0.396   |
| LLaVA-Next-Video              | 4.19        | 4.07   | 4.39    | 4.04     | 0.250  | 0.249   | 0.395   |
| ShareGPT4Video                | 4.24        | 4.13   | 4.35    | 4.09     | 0.192  | 0.205   | 0.394   |
| LLaMA-VID                     | 3.95        | 4.01   | 4.22    | 3.71     | 0.195  | 0.231   | 0.339   |
| VideoLLaMA2                   | 4.17        | 4.02   | 4.47    | 3.93     | 0.253  | 0.266   | 0.344   |
| PLLaVA                        | 4.21        | 4.15   | 4.37    | 4.08     | 0.268  | 0.250   | 0.393   |
| ST-LLM                        | 4.00        | 3.98   | 4.31    | 3.94     | 0.213  | 0.238   | 0.321   |
| Tarsier                       | 3.59        | 3.50   | 4.07    | 3.41     | 0.143  | 0.185   | 0.415   |
| LLaVA-OneVision              | 3.68        | 3.47   | 4.10    | 3.42     | 0.115  | 0.165   | 0.379   |
| FaceTrack-MM                 | 4.42        | 4.30   | 4.60    | 4.26     | 0.418  | 0.473   | 0.483   |
| Qwen2-VL-72B                 | 4.28        | 4.14   | 4.55    | 4.08     | 0.241  | 0.314   | 0.449   |
| Qwen2-VL-7B                  | 4.23        | 4.16   | 4.52    | 4.02     | 0.204  | 0.233   | 0.422   |
| Qwen2-VL-2B                  | 4.01        | 3.98   | 4.37    | 3.88     | 0.202  | 0.221   | 0.406   |
| Claude3.5-Sonnet             | 4.13        | 4.01   | 4.49    | 4.05     | 0.243  | 0.228   | 0.442   |
| **Omni-modality large language model** |      |        |         |          |        |         |         |
| GPT4-O                       | 4.22        | 3.97   | 4.48    | 3.90     | 0.264  | 0.213   | 0.432   |
| VITA                         | 3.98        | 3.74   | 4.11    | 3.59     | 0.191  | 0.224   | 0.366   |
| InternLM-XComposer-2.5-OL    | 3.91        | 3.70   | 4.12    | 3.54     | 0.113  | 0.164   | 0.382   |
| **HumanOmni**                    | **4.58**    | **4.41**| **4.70**| **4.41** | 0.412  | 0.468   | **0.523**|

- Action and Pose Understanding:

| Method                          | Action Sequence  | Unexpected Action  | Action Antonym  | Object Interaction  | Action Count  | Fine-grained Action  | Avg  |
|----------------------------------|-----|-----|-----|-----|-----|-----|------|
| **Vision large language model** |     |     |     |     |     |     |      |
| Otter-V                       | 23.0| 29.5| 27.5| 28.0| 26.0| 27.0| 26.8 |
| mPLUG-Owl-V                   | 22.0| 29.0| 34.0| 27.0| 31.5| 29.0| 28.8 |
| Video-LLaMA                   | 27.5| 39.0| 51.0| 40.5| 34.0| 29.0| 36.8 |
| LLaMA-Adapter                 | 23.0| 33.0| 51.0| 32.5| 29.0| 30.0| 33.1 |
| Video-ChatGPT                 | 23.5| 26.5| 62.0| 28.0| 30.5| 22.5| 32.2 |
| VideoChat                     | 33.5| 40.5| 56.0| 40.5| 35.0| 33.5| 39.8 |
| VideoChat2                    | 75.5| 60.5| 83.5| 74.5| 37.0| 50.5| 63.6 |
| ST-LLM                        | 66.0| 58.5| 84.0| 73.5| 36.5| 44.0| 60.4 |
| PLLaVA                        | 58.0| 61.0| 55.5| 61.0| 39.5| 41.0| 52.6 |
| VideoLLaMB                   | 54.5| 52.0| 86.5| 58.5| 40.5| 44.5| 56.1 |
| Qwen2-VL-72B*                | 51.5| 82.0| 93.5| 81.5| 48.5| 49.0| 67.7 |
| Qwen2-VL-7B*                 | 73.5| 80.0| 79.0| 78.5| 46.0| 49.0| 67.7 |
| Qwen2-VL-2B*                 | 77.5| 76.5| 76.5| 77.5| 50.0| 47.5| 67.6 |
| GPT-4V                       | 55.5| 63.5| 72.0| 59.0| 39.0| 47.5| 56.1 |
| **Omni-modality large language model** |     |     |     |     |     |     |      |
| VITA                         | 58.0| 81.5| 73.5| 61.5| 45.5| 42.0| 60.3 |
| InternLM-XComposer-2.5-OL    | 84.5| 81.0| 75.0| 79.5| 60.5| 46.0| 71.1 |
| **HumanOmni**                    | 70.0| 78.0| 92.5| 80.5| 65.5| 49.0| **72.6** |

## 📺  Demo
https://github.com/user-attachments/assets/ecf07e98-f448-4cb6-a124-6e2a2885e581
```
User:What's the major emotion?
HumanOmni:angry.

User:What's the major facial expression?
HumanOmni:frown.

User:Desribe the woman.
HumanOmni:The woman has long brown hair and is wearing a pink top with a dark cardigan.

User:Please describe the major action in detail.
HumanOmni:The woman is frowning and shaking her head, indicating that she is angry. She is also gesturing with her hands as if she is explaining something or emphasizing a point.

User:What did the woman say?
HumanOmni:The woman said "Of course we object! I object, Steve!"
```

https://github.com/user-attachments/assets/0f0d9c5c-6e1d-473f-8c6b-819ebdc8aae8
```
User:What's the major emotion?
HumanOmni:happy.

User:What's the major facial expression?
HumanOmni:smile.

User:Desribe the woman in detail.
HumanOmni:The woman in the video has blonde hair, is wearing a black top, and is smiling.

User:What did the woman say?
HumanOmni:The woman said "Dogs are sitting by the door."
```

[https://humanmllm.github.io/HumanOmni/chinese_video.mp4](https://github.com/user-attachments/assets/65b49d37-7213-452d-a492-cfddbbc8b41d)
```
User:描述一下视频中男子的外貌穿着.
HumanOmni:视频中的男性有短的黑发，配戴圆框眼镜。他穿了一件棕色的毛大衣，内里是白色的.

User:视频中男子说了什么内容？
HumanOmni:男子说他穿什么颜色的衣服，现在处于什么环境，今天的天气怎么样.

User:视频中男子有什么情绪吗？
HumanOmni:没有情绪。
```

## 🛠️ Environment Setup

To set up the recommended environment for HumanOmni, follow these instructions:

### Recommended Environment
- **Python**: >=3.10
- **CUDA**: >=12.1
- **PyTorch**: >=2.2 (with CUDA support)
- **Transformers**: >=4.45
- **Accelerate**: >=0.30.1

Or you can quickly set up the environment as follows:

```
git clone https://github.com/HumanMLLM/HumanOmni
cd HumanOmni
conda create -n humanOmni python=3.10 -y
conda activate humanOmni
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
## 🧠 Training on Custom Dataset
### Data Preparation
An example json file of the training data:
```
[
    {
        "video": "human/DFEW/videos/1.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\n<audio>\nAs an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?\nfear ,angry ,surprise ,happy ,neutral ,sad ,disgust"
            },
            {
                "from": "gpt",
                "value": "sad"
            }
        ],
    },
    {
        "video": "human/DFEW/videos/1.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\n<audio>\nAs an emotional recognition expert, in the video, when the characters display their emotions, which predominant feeling is most clearly expressed?\nfear ,disgust ,happy ,sad ,surprise"
            },
            {
                "from": "gpt",
                "value": "sad"
            }
        ],
    },
  ...
]
```

### Multi-Modal SFT
- Download the required weights: (1) [HumanOmni-7B-Video](https://modelscope.cn/models/iic/HumanOmni-7B-Video) (2) [HumanOmni-7B-Audio](https://modelscope.cn/models/iic/HumanOmni-7B-Audio)
- scripts/train/finetune_humanomni.sh Loading the weights and the prepared dataset.
- bash scripts/train/finetune_humanomni.sh

## 🔍 Inference
We provide inference.py for singe video inference. 
 - video + audio 
```
python inference.py --modal video_audio \
  --model_path ./HumanOmni_7B \
  --video_path video.mp4 \
  --instruct "Describe this video."
```
 - only video 
```
python inference.py --modal video \
  --model_path ./HumanOmni_7B \
  --video_path video.mp4 \
  --instruct "Describe this video."
```
- only audio
```
python inference.py --modal audio \
  --model_path ./HumanOmni_7B \
  --video_path video.mp4 \
  --instruct "Describe this video."
```

## 🤝 Related Work
- [LLaVA-Octopus: Unlocking Instruction-Driven Adaptive Projector Fusion for Video Understanding](https://arxiv.org/abs/2501.05067)
- [Omni-Emotion: Extending Video MLLM with Detailed Face and Audio Modeling for Multimodal Emotion Analysis](https://arxiv.org/abs/2501.09502)
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)

## 📚 Citation
If you find our work helpful, feel free to give us a cite.
```
@article{zhao2025humanomni,
  title={HumanOmni: A Large Vision-Speech Language Model for Human-Centric Video Understanding},
  author={Zhao, Jiaxing and Yang, Qize and Peng, Yixing and Bai, Detao and Yao, Shimin and Sun, Boyuan and Chen, Xiang and Fu, Shenghao and Wei, Xihan and Bo, Liefeng and others},
  journal={arXiv preprint arXiv:2501.15111},
  year={2025}
}
```
