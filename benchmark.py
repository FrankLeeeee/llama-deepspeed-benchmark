import argparse

import torch
import deepspeed
from attn import SUPPORT_FLASH, replace_xformers
from data_utils import RandomDataset
from model_utils import format_numel_str, get_model_numel
from performance_evaluator import PerformanceEvaluator
from tqdm import tqdm
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import torch.distributed as dist
import json
from deepspeed.accelerator import get_accelerator

try:
    import torch_npu
    import deepspeed_npu
except:
    pass



# ==============================
# Constants
# ==============================

MODEL_CONFIGS = {
    "7b": LlamaConfig(max_position_embeddings=4096),
    "13b": LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        max_position_embeddings=4096,
    ),
    "70b": LlamaConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        max_position_embeddings=4096,
        num_key_value_heads=8,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="7b", type=str, help="model type")
    parser.add_argument("-s", "--num_steps", type=int, default=5, help="Number of steps to run")
    parser.add_argument("-i", "--ignore_steps", type=int, default=2, help="Number of steps to ignore")
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("-l", "--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument(
        "-w", "--warmup_ratio", type=float, default=0.8, help="warm up ratio of non-model data. Only for gemini-auto"
    )
    parser.add_argument('--local_rank', type=int, default=-1, help="local rank for distributed training on gpus")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    args = parse_args()
    deepspeed.init_distributed()
    print("initialized distributed")

    # get config
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)
    batch_size_per_gpu = ds_config['train_micro_batch_size_per_gpu']

    # ==============================
    # Initialize Dataset and Dataloader
    # ==============================
    dp_size = dist.get_world_size()
    config = MODEL_CONFIGS[args.model]
    dataset = RandomDataset(
        num_samples=batch_size_per_gpu * args.num_steps * dp_size, max_length=args.max_length, vocab_size=config.vocab_size
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_per_gpu,
        num_workers=8, pin_memory=True, sampler=sampler)
    print("built dataloader")

    # ==============================
    # Initialize Model and Optimizer
    # ==============================
    with deepspeed.zero.Init(remote_device=None,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=ds_config['zero_optimization']['stage'] == 3
                             ):
        model = LlamaForCausalLM(config)
    print("built model")

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        print("enabled gradient checkpointing")

    # if args.xformers:
    #     assert SUPPORT_FLASH, "Use flash attention while xfomers is not installed"
    #     replace_xformers(model)

    # ==============================
    # Initialize Model and Optimizer
    # ==============================
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        )
    print("initialized deepspeed")

    model_numel = sum(p.numel() for p in model_engine.module.parameters())
    performance_evaluator = PerformanceEvaluator(
        model_numel=model_numel,
        num_layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size,
        vocab_size=model.config.vocab_size,
        dp_world_size=dp_size,
        enable_grad_checkpoint=args.grad_checkpoint,
        ignore_steps=args.ignore_steps,
    )


    for step, data in tqdm(enumerate(dataloader), desc="Step", disable=dist.get_rank() != 0):
        performance_evaluator.on_step_start(step)
        model_engine.zero_grad()
        data = {
            k: v.to(get_accelerator().current_device())
            for k, v in data.items()
        }
        outputs = model_engine(**data)
        loss = outputs[0]
        model_engine.backward(loss)
        model_engine.step()
        
        performance_evaluator.on_step_end(input_ids=torch.empty(batch_size_per_gpu, args.max_length))

    performance_evaluator.on_fit_end()
    print(f"Max CUDA memory usage: {get_accelerator().max_memory_allocated()/1024**2:.2f} MB")


if __name__ == "__main__":
    main()
