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
from tqdm import tqdm

# disable wandb for this script
# os.environ['WANDB_MODE'] = 'disabled'

instruction_tuning = True
save_interval = 1030
log_interval = 1

# constants
multi_dialogue = 'multi-dialogue'
# Hyperparameters
learning_rate = 3e-4
batch_size = 64
micro_batch_size = 1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_epochs = 10
max_iters = save_interval * max_epochs // micro_batch_size  # it seems that alpaca is obtained after 3 epochs, but lima needs more
weight_decay = 0.0
max_seq_length = 1024  # see scripts/prepare_lima.py
lora_r = 8
lora_alpha = 16
lora_dropout = 0.1

def main(
    data_dir: str = "data/deita-6k-v0", 
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    out_dir: str = None,
    smooth:float = 0.0,
):  
    # recognize the dataset name from the data_dir
    # and reset hyperparameters as well
    dataset_name = data_dir.split('/')[-1]
    reset_hyperparameters__(dataset_name)
    __running_tag = formulate_specific_tag__(dataset_name, smooth)
    

    if out_dir is None:
        out_dir = f"out/lora/{dataset_name}/{__running_tag.replace(' ', '_')}"

    # innitialize wandb monitor process
    wandb.init(project='lima-sft', name=__running_tag)  

    fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-true")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data = load_datasets(data_dir=data_dir)
    if multi_dialogue in __running_tag and 'lima' in dataset_name:
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
    awarer = UncertaintyAware()

    pbar = tqdm(range(max_iters))
    for iter_num in pbar:

        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        return_dict = get_batch(fabric, train_data)

        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
            logits = model(return_dict['input_ids'])
            loss_constraits = awarer.get_value(logits, return_dict) if smoothing != 0.0 else 0.0
            loss = loss_fn(logits, return_dict['labels'], smoothing=loss_constraits)
            fabric.backward(loss / gradient_accumulation_iters)
            accumulated_loss += loss.item()

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            wandb.log({"loss": accumulated_loss / gradient_accumulation_iters})
            wandb.log({"smooth_value": awarer.last()})
            accumulated_loss = 0.0

        if (iter_num + 1) % save_interval == 0:
            fabric.barrier()
            print(f"Saving LoRA weights to {out_dir}")
            # We are only saving the LoRA weights
            # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
            checkpoint = lora_state_dict(model)
            fabric.save(os.path.join(out_dir, f"iter-{iter_num + 1:06d}-ckpt.pth"), checkpoint)

        dt = time.time() - t0

        if iter_num % log_interval == 0:
            __log_info = f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms"
            if smoothing != 0.0:
                __log_info += f", smooth_value {loss_constraits[0]:.4f}"
            pbar.set_description(__log_info)
    
    with open(os.path.join(out_dir, 'smooth_values.json'), 'w') as f:
        import json
        json.dump(awarer.all(), f)


class UncertaintyAware:
    def __init__(self, target_avg=0.1):
        assert 0.0 < target_avg < 1.0
        self.target_avg = target_avg
        self.move_avg = MovingAverage()
        self.__result_move_avg = MovingAverage()


    def __ppl_cal(self, logits, labels):
        """Calculate the perplexity of the logits."""
        # (batch_size, seq_len, vocab_size), (batch_size, seq_len)
        log_preds = F.log_softmax(logits, dim=2)
        sum_log_preds =  torch.gather(log_preds, 2, labels.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        _mask = (labels != -1).to(torch.float).sum(dim=1)
        ppl = -sum_log_preds / _mask

        return ppl.cpu().detach()

    def get_value(self, logits, return_dict):
        """Calculate the uncertainty based on inputs PPL."""
        ppl = self.__ppl_cal(logits, return_dict['labels'])
        ppl_floats = ppl.view(-1).numpy().tolist()
        for i, ppl_f in enumerate(ppl_floats):
            self.move_avg.update(ppl_f)
        # TODO: the methodology of uncertainty-aware is not clear
        factors = self.move_avg.get() / ppl.view(-1).numpy()
        # intuitive understanding: the higher the uncertainty, the less the smooth value in cross entropy
        smooth_values = [self.target_avg * factor for factor in factors.tolist()]
        smooth_values = np.clip(np.array(smooth_values), 0.0, 0.99).tolist()

        # record the smooth values for logging
        for i, smooth_value in enumerate(smooth_values):
            self.__result_move_avg.update(smooth_value)
        return smooth_values


    def final(self):
        # report the average smooth value
        return self.__result_move_avg.get()
    
    def last(self):
        return self.__result_move_avg.values[-1]
    
    def all(self):
        return self.__result_move_avg.values


def label_smooth(labels, classes, smoothing=0.1):
    """
    Applies label smoothing to the given labels.
    
    Args:
        labels (Tensor): A tensor containing the labels.
        classes (int): Total number of classes.
        smoothing (float): Smoothing factor.
        
    Returns:
        Tensor: A new tensor with smoothed labels.
    """
    original_device = labels.device
    if isinstance(smoothing, list):
        assert len(smoothing) == labels.size(0)

        labels_copy = labels.clone()
        labels_copy[labels_copy == -1] = 0

        smoothed_list = []
        for i, smooth_value in enumerate(smoothing):
            confidence = 1.0 - smooth_value
            smooth_label = torch.full(size=(labels.size(1), classes), fill_value=smooth_value / (classes - 1)).to(labels.device)
            smooth_label.scatter_(1, labels_copy[i].unsqueeze(-1), confidence)
            smoothed_list.append(smooth_label)
        smooth_label = torch.stack(smoothed_list)

    else:
        # Create a tensor with smoothing/num_classes for each label
        confidence = 1.0 - smoothing

        # offload for saving some GPU memory
        labels = labels.cpu()
        smooth_label = torch.full(size=(labels.size(0), labels.size(1), classes), fill_value=smoothing / (classes - 1)).to(labels.device)
        # set labels = 0 where labels == -1, in case of CUDA insertion error
        labels_copy = labels.clone()
        labels_copy[labels_copy == -1] = 0
        smooth_label.scatter_(2, labels_copy.unsqueeze(-1), confidence)
    
    return smooth_label.to(original_device)


def loss_fn(logits, targets_, smoothing=0.0):
    # TODO: support mask inputs for label smoothing
    targets = label_smooth(targets_, logits.size(-1), smoothing=smoothing)
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:, :].contiguous()

    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1, targets.size(-1))

    real_mask = (targets_ != -1)
    mask = real_mask.unsqueeze(-1).expand(-1, -1, logits.size(-1))[..., 1:, :]
    mask = mask.view(-1, logits.size(-1)).float()

    log_preds = F.log_softmax(logits, dim=1)
    log_preds = log_preds * mask
    loss = -torch.sum(log_preds * targets) / real_mask.float().sum()
    # import pdb; pdb.set_trace()
    return loss


def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]
    smooth_values = None
    if "smooth_value" in data[0].keys():
        smooth_values = [data[i]["smooth_value"] for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return {"input_ids": x, "labels": y, "smooth_values": smooth_values}


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    return train_data


class MovingAverage:
    def __init__(self):
        self.values = []
        self.tot = 0.0
    
    def update(self, value):
        assert isinstance(value, float)
        self.values.append(value)
        self.tot += value
    
    def get(self):
        return self.tot / len(self.values)


def reset_hyperparameters__(dataset_name):
    global save_interval, max_iters, max_epochs, warmup_iters

    assert dataset_name in ['lima', 'deita-6k-v0']
    if dataset_name == 'lima':
        save_interval = 1030
        max_epochs = 10
        # it seems that alpaca is obtained after 3 epochs, but lima needs more
        max_iters = save_interval * max_epochs // micro_batch_size
        warmup_iters = int(0.1 * max_iters)
    elif dataset_name == 'deita-6k-v0':
        # follow the paper setting: 
        # WHAT MAKES GOOD DATA FOR ALIGNMENT? A COMPREHENSIVE STUDY OF AUTOMATIC DATA SELECTION IN INSTRUCTION TUNING
        # (https://arxiv.org/pdf/2312.15685.pdf)
        max_epochs = 6
        save_interval = 6000
        max_iters = save_interval * max_epochs // micro_batch_size
        warmup_iters = int(0.1 * max_iters)


def formulate_specific_tag__(dataset_name, smooth):
    __running_tag=f'{"ual" if smooth else "sft"}_'\
        f'{dataset_name}_'\
        f'lora_sctx-{max_seq_length}_micro{micro_batch_size}_'\
        f'epoch{max_epochs}'\
        f'{("" if smooth == 0.0 else f"_ls-{smooth:0.2f}")} '+\
        datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    __running_tag = __running_tag.replace('micro', f'{multi_dialogue}-micro')
    return __running_tag


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    
    from jsonargparse.cli import CLI

    CLI(main)
