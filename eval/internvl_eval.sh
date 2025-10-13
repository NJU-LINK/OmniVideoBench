export CUDA_VISIBLE_DEVICES=0,1
source /cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/env/internvl/bin/activate

python internvl_eval.py \
    --data_json_file "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/final_data/out_2.json" \
    --video_dir "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/omni_videos_v1" \
    --output_file "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/eval_results/internvl_38b_64frames_out_2.json" \
    --model_path "/cpfs01/user/liujiaheng/workspace/caoruili/omni-videos-lcr/code/models/InternVL3_5-38B" \
    --max_duration 6000 \
    --num_segments 64 \
    --max_num 1


# model_path 模型路径, 改后缀 38B和9b
# num_segments每个视频提取的帧数
# max_num 每帧的最大tile数量,视频默认为 1