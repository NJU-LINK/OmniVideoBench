set -x
export HF_HOME=/mnt/data/jiaxing.zjx/cache/huggingface/
export HF_ENDPOINT=http://hf-mirror.com
EVAL_DATA_DIR=/mnt/data/qize.yqz/datasets/video_eval
OUTPUT_DIR=eval_output_qwen_noaudio
CKPT=./HumanOmni_7B

CKPT_NAME=$(echo $CKPT | rev | cut -d'/' -f1 | rev)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/mvbench/answers/${CKPT_NAME}/merge.json

# judge if the number of json lines is 0
if [ ! -f "$output_file" ] || [ $(cat "$output_file" | wc -l) -eq 0 ]; then
    rm -f ${OUTPUT_DIR}/mvbench/answers/${CKPT_NAME}/*.json
fi

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 humanomni/eval/inference_video_mcqa_mvbench.py \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/MVBench/video \
            --question-file ${EVAL_DATA_DIR}/MVBench/json \
            --answer-file ${OUTPUT_DIR}/mvbench/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/mvbench/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi

python3 humanomni/eval/eval_video_mcqa_mvbench.py \
    --pred_path ${output_file} \
