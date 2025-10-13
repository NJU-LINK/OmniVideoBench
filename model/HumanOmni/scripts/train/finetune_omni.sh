#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=128
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
echo $GRADIENT_ACCUMULATION_STEPS

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=humanomniqwen2_siglip
export HF_HOME=/mnt/data/jiaxing.zjx/cache/huggingface/
export HF_ENDPOINT=http://hf-mirror.com
RUN_NAME=HumanOmni
DATA_DIR=/mnt/data/jiaxing.zjx/datasets/Video-LLaVA/
OUTP_DIR=work_dirs

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    humanomni/train_flash_attn.py \
    --deepspeed scripts/zero3.json \
    --model_type HumanOmni_qwen2 \
    --model_path /mnt/data/jiaxing.zjx/code/HumanOmni/HumanOmni_7B_Video/ \
    --vision_tower google/siglip-so400m-patch14-384 \
    --audio_tower openai/whisper-large-v3 \
    --mm_projector_type  all_in_one \
    --mm_tunable_parts "mm_mlp_adapter,audio_projector,mm_language_model" \
    --pretrain_audio_mlp_adapter /mnt/data/jiaxing.zjx/code/HumanOmni/HumanOmni_7B_Audio/audio_projector.bin \
    --data_path   ./yamls/oryx_audio.yaml \
    --data_folder / \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --num_frames 32 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/finetune_${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 99 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --mm_use_x_start_end True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --run_name $RUN_NAME \
