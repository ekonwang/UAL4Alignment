"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import sys
from pathlib import Path
import os
import datetime
import time

import lightning as L
import numpy as np
import torch
import wandb
import torch.nn.functional as F

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import LLaMA, LLaMAConfig
from scripts.prepare_alpaca import generate_prompt
from finetune.llama import loss_fn, get_batch, load_datasets


instruction_tuning = True
save_interval = 20
log_interval = 1

# Hyperparameters
learning_rate = 3e-4
batch_size = 64
micro_batch_size = 1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_epochs = 15
max_iters = 1030 * max_epochs // micro_batch_size  # it seems that alpaca is obtained after 3 epochs, but lima needs more
weight_decay = 0.0
max_seq_length = 512  # see scripts/prepare_lima.py
lora_r = 8
lora_alpha = 16
lora_dropout = 0.1
warmup_iters = 100
smooth = 0.1


def main(
    data_dir: str = "data/lima", 
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    out_dir: str = None,
):
    
    assert "ls" in data_dir

    tag=f'sft_ada-lima_lora_sctx-{max_seq_length}' + \
        ("" if smooth == 0.0 else f"_{data_dir.split('/')[-1]} ") + \
        datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    multi_dialogue = 'multi-dialogue'
    tag=tag.replace('micro', f'{multi_dialogue}-micro')

    wandb.init(project='lima-sft', name=tag)
    if out_dir is None:
        out_dir = f"out/lora/lima/{tag.replace(' ', '_')}"

    fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-true")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data = load_datasets(data_dir=data_dir)
    if multi_dialogue in tag:
        assert len(train_data) == 1030  # assert the 30 multi-turn dialogues are included
    config = LLaMAConfig.from_name("7B")
    config.block_size = max_seq_length

    checkpoint = torch.load(pretrained_path)

    with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        model = LLaMA(config)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        model.load_state_dict(checkpoint, strict=False)
    
    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, out_dir, smoothing=smooth)
    wandb.finish()

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(out_dir, "lit-llama-lora-finetuned.pth"), checkpoint)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    out_dir: str,
    smoothing: float = 0.0
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    accumulated_loss = 0.0

    print()
    print('+++++++++++++ Training +++++++++++++')
    print('+                                  +')
    print('+          Adaptive LIMA           +')
    print('+                                  +')
    print('++++++++++++++++++++++++++++++++++++')
    print()

    for iter_num in range(max_iters):

        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_dict = get_batch(fabric, train_data)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
            logits = model(input_dict["input_ids"])
            loss = loss_fn(logits, input_dict["labels"], smoothing=input_dict["smooth_values"])
            fabric.backward(loss / gradient_accumulation_iters)
            accumulated_loss += loss.item()

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            wandb.log({"loss": accumulated_loss / gradient_accumulation_iters})
            accumulated_loss = 0.0

            if step_count % save_interval == 0:
                fabric.barrier()
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model)
                fabric.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"), checkpoint)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    
    from jsonargparse.cli import CLI

    CLI(main)
