import torch
import wandb
import time
import os
import sys
import json
import datetime
import lightning as L

from pathlib import Path
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import LLaMA, LLaMAConfig
from utils import loss_fn, print_trainable_parameters, make_score_dist

# disable wandb
os.environ['WANDB_MODE'] = 'disabled'

lora_r = 8
lora_alpha = 16
lora_dropout = 0.1

lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=[
        "q_proj", # change q, v attention is enough
        "v_proj",
    ],
    bias="none",
    lora_dropout=lora_dropout,
    task_type="CAUSAL_LM",
)

train_config = {
    'learning_rate': 3e-4,
    'batch_size': 64,
    'micro_batch_size': 1,
    'gradient_accumulation_iters': 64,
    'max_epochs': 10,
    'log_interval': 1,
    'save_iterval': 0, # need to be reset
    'max_iters': 0, # need to be reset
    'warmup_iters': 0, # need to be reset
    'weight_decay': 0.0,
    'max_seq_length': 1024, # by default
}

model_configs = {
    "llama2-7b": "./checkpoints/lit-llama/7B/lit-llama.pth",
    "mistral-7b": "mistralai/Mistral-7B-v0.1"
}


def main(
    model_nickname: str = "mistral-7b",
    data_path: str = "data/sharegpt-6k",
    out_dir: str = None,
    smooth: float = 0.0,
    smooth_strategy: str = "equal", 
):
    # check parameters
    train_config['smooth_strategy'] = smooth_strategy
    train_config['model_nickname'] = model_nickname
    check_hyperparameters__(train_config)

    # load dataset
    dataset = load_datasets(data_path, train_config, smooth=smooth)
    # reset hyperprameters
    reset_hyperparameters__(dataset, train_config)

    # accelerator, could lightning save the world?
    fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-true")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)
    
    # load model and tokenizer
    model, tokenizer = load_model(train_config, fabric)

    # running identifier
    dataset_name = data_path.split('/')[-1]
    __running_tag = formulate_specific_tag__(dataset_name, smooth, train_config)
    if out_dir is None:
        out_dir = f"out/lora/{dataset_name}/{__running_tag.split(' ')[0]}"
    os.makedirs(out_dir, exist_ok=True)
    # upload to wandb
    wandb.init(project='lima-sft', name=__running_tag)  

    print_trainable_parameters(model)
    # Done: check if the Lora parameter could be saved
    # Done: check if the Lora parameters could be merged and unmerged
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['learning_rate'])
    model, optimizer = fabric.setup(model, optimizer)

    # training
    train(model, fabric, optimizer, dataset, 
          out_dir, train_config, smoothing=smooth)
    wandb.finish()

    # save the last ckpt
    checkpoint = lora_state_dict(model)
    torch.save(checkpoint, os.path.join(out_dir, "lit-llama-lora-finetuned.pth"))


def train(
    model,
    fabric: L.Fabric,
    optimizer: torch.optim.Optimizer,
    train_data: list,
    out_dir: str,
    config: dict,
    smoothing: float = 0.0
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    accumulated_loss = 0.0

    pbar = tqdm(range(config['max_iters']))
    for iter_num in pbar:

        if step_count <= config['warmup_iters']:
            # linear warmup
            lr = config['learning_rate'] * step_count / config['warmup_iters']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        return_dict = get_batch(fabric, train_data, config)

        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % config['gradient_accumulation_iters'] != 0)):
            logits = model(return_dict['input_ids']).logits
            loss = loss_fn(logits, return_dict['labels'], smoothing=smoothing)

            fabric.backward(loss / config['gradient_accumulation_iters'])
            accumulated_loss += loss.item()

        if (iter_num + 1) % config['gradient_accumulation_iters'] == 0:
            fabric.barrier()
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            wandb.log({"loss": accumulated_loss / config['gradient_accumulation_iters']})
            accumulated_loss = 0.0

        if (iter_num + 1) % config['save_interval'] == 0:
            fabric.print(f"Saving LoRA weights to {out_dir}")
            # We are only saving the LoRA weights
            # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
            checkpoint = lora_state_dict(model)
            fabric.save(os.path.join(out_dir, f"iter-{iter_num + 1:06d}-ckpt.pth"), checkpoint)

        dt = time.time() - t0

        if iter_num % config['log_interval'] == 0:
            __log_info = f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms"
            pbar.set_description(__log_info)


def load_model(fabric: L.Fabric, config: dict):
    m_nick = train_config['model_nickname']
    model_name_or_path = train_config['model_name_or_path']
    max_seq_length = train_config['max_seq_length']

    if m_nick == 'mistral-7b':
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # make lora model
        model = get_peft_model(model, lora_config)
    elif m_nick == 'llama2-7b':
        config = LLaMAConfig.from_name("7B")
        config.block_size = max_seq_length
        checkpoint = torch.load(model_name_or_path)

        with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA(config)
            # strict=False because missing keys due to LoRA weights not contained in checkpoint state
            model.load_state_dict(checkpoint, strict=False)
        mark_only_lora_as_trainable(model)    
    return model, tokenizer


def load_datasets(data_path, config, smooth):
    nickname = config['model_nickname']
    if nickname == 'llama2-7b':
        train_data = torch.load(os.path.join(data_path, f"train.pt"))
    elif nickname == 'mistral-7b':
        train_data = torch.load(os.path.join(data_path, f"train_Mistral-7B-v0.1.pt"))

    score_path = Path(data_path) / 'score.json'
    with open(score_path) as f:
        scores = json.load(f)
        assert len(scores) == len(train_set)
    scores = make_score_dist(scores, target_mean=smooth)
    train_set = [dict(sample, smooth_value=score) for sample, score in zip(train_set, scores)]

    return train_data


def lora_state_dict(model, bias: str = 'none'):
    """Return state_dict with weights of LoRA's A and B matrices and with biases depending on the `bias` value.
    """
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}


def get_batch(fabric: L.Fabric, data: list, config: dict):
    ix = torch.randint(len(data), (config['micro_batch_size'],))

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
    x = x[..., :config['max_seq_length']]
    y = y[..., :config['max_seq_length']]
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return {"input_ids": x, "labels": y, "smooth_values": smooth_values}


def check_hyperparameters__(config):
    assert train_config['model_nickname'] in model_configs.keys()
    train_config['model_name_or_path'] = model_configs[train_config['model_nickname']]


def reset_hyperparameters__(dataset, config):
    model_nickname = config['model_nickname']
    if model_nickname == 'mistral-7b':
        config['max_seq_length'] = 768

    config['save_interval'] = len(dataset)
    config['max_iters'] = config['save_interval'] * config['max_epochs'] // config['micro_batch_size']
    config['warmup_iters'] = int(0.1 * config['max_iters'])


def formulate_specific_tag__(dataset_name, smooth, config):
    nickname = config['model_nickname']
    max_epochs = config['max_epochs']
    smooth_strategy = config['smooth_strategy']
    micro_batch_size = config['micro_batch_size']
    max_seq_length = config['max_seq_length']

    label_smooth_tag = ("" if smooth == 0.0 else f"_{smooth_strategy}-ls-{smooth:0.2f}")

    __running_tag=f'{nickname}/sft_'\
        f'{dataset_name}_'\
        f'lora_sctx-{max_seq_length}_micro{micro_batch_size}_'\
        f'epoch{max_epochs}'\
        f'{label_smooth_tag} '+\
        datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    return __running_tag


def load_lora_ckpt_from_disk_to_model__(lora_path, model):
    lora_ckpt = torch.load(lora_path)
    model.load_state_dict(lora_ckpt, strict=False)
    # TODO: check sanity 
    # merge LoRA parameters into model to accelerate inference.
    model.merge_adapter()
    # NOTES: use `unmerge_adapter` to separate LoRA parameters from model.

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    from jsonargparse.cli import CLI

    CLI(main)
