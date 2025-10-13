import os
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from accererate import Accereator


accelerator = Accelerator(mixed_precision="bf16")



def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()

def setup_distributed(rank, world_size):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def load_model_and_processor(model_name: str, rank: int):
    """Load and initialize the Qwen model and processor with distributed setup."""
    print(f"Loading model and processor on rank {rank}...")
    
    # Load model on specific GPU
    model = Qwen2ForCausalLM_RingAttn.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        device_map=accelerator.device
    )   
    
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    return model, processor

def split_sequence(inputs, rank, world_size):
    """Split input sequence across multiple GPUs for ring attention."""
    split_inputs = {}
    start_indices = {}
    end_indices = {}
    
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.dim() > 1:
            # 为每个张量计算其自己的序列长度
            seq_len = value.shape[1]
            chunk_size = seq_len // world_size
            start_idx = rank * chunk_size
            
            # Handle remainder for last GPU
            if rank == world_size - 1:
                end_idx = seq_len
            else:
                end_idx = start_idx + chunk_size
            
            split_inputs[key] = value[:, start_idx:end_idx]
            start_indices[key] = start_idx
            end_indices[key] = end_idx
        else:
            split_inputs[key] = value
            start_indices[key] = 0
            end_indices[key] = 0
    
    # 返回文本序列的索引作为主要参考
    main_start = start_indices.get('input_ids', 0)
    main_end = end_indices.get('input_ids', 0)
    
    return split_inputs, main_start, main_end


def gather_ring_attention_outputs(local_output,world_size):
    """Gather outputs from all GPUs in ring attention setup."""
    gathered_outputs=[torch.zeros_like(local_output) for _ in range(world_size)]
    dist.all_gather(gathered_outputs,local_output)
    full_output=torch.cat(gathered_outputs,dim=1)
    return full_output


