import os
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import deepspeed
from torch.distributed import init_process_group, get_rank, get_world_size
from deepspeed import comm
# from ulysses import load_model


def init_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def load_model_and_processor_with_deepspeed(model_name: str, local_rank: int):
    """Load and initialize the Qwen model and processor with DeepSpeed."""
    print(f"Loading model and processor on rank {get_rank() if torch.distributed.is_initialized() else 0}...")
    
    # Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    
    # Load model
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    
    # DeepSpeed configuration for Ulysses sequence parallelism
    ds_config = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 0  # No ZeRO for inference
        },
        "ulysses": {
            "enabled": True,
            "degree": get_world_size() if torch.distributed.is_initialized() else 1
        },
        "sequence_parallel_size": get_world_size() if torch.distributed.is_initialized() else 1
    }
    
    # Initialize DeepSpeed engine
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )

    return model_engine, processor

# def load_model_and_processor_with_deepspeed(model_path, local_rank):
#     model, tokenizer = load_model(
#         model_path,
#         device=f"cuda:{local_rank}",
#         dtype="fp16",
#         deepspeed=True,                     
#         ring_attention=True,               
#         use_flash_attention_2=True,       
#         rope_scaling={"type": "linear", "factor": 1.0},
#     )
#     processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    
#     return model, tokenizer
