export CUDA_VISIBLE_DEVICES=1
# export DEEPSPEED_DISABLE=1
export TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p /tmp/triton_cache
uv run eval.py