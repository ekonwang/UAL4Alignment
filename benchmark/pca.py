import sys
import os
import json
import time
import warnings
import datetime
from pathlib import Path

import lightning as L
import torch
from datasets import load_dataset
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup
from scripts.prepare_lima import generate_prompt

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05


def main(
    lora_path: str = None,
    pretrained_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    data_dir: str = "./data/lima",
    output_file: Path = Path("./out/benchmark/pca_analysis/llama2-7b-lima-test-features.pt"),
    data_split: str = "test",
) -> None:
    
    # lora path could be empty, which means the pretrained model
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA.from_name(name)

            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned lora weights
            if lora_checkpoint is not None:
                model.load_state_dict(lora_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)
    tokenizer = Tokenizer(tokenizer_path)

    collected_transformer_features = list()
    encoded = None
    test_dataset = load_dataset(data_dir, split=data_split)

    def transformer_hook(model, input, output):
        # the output is the results of the last RMS norm layer, size is (B, T, embed_size)
        embed_size = output.size(-1)
        features = output.clone().cpu().detach()[...,:-1,:].view(-1, embed_size)
        labels = encoded[...,1:].view(-1).tolist()

        assert len(labels) == features.size(0)
        
        for label, feature in zip(labels, features):
            collected_transformer_features.append((label, feature))
        print(len(label))
        import pdb; pdb.set_trace()

    hook = model.transformer.register_forward_hook(transformer_hook)

    for sample in tqdm(test_dataset):
        encoded = sample['input_ids'].view(1, -1)
        _ = model(encoded)
        import pdb; pdb.set_trace()
    
    hook.remove()
    torch.save(collected_transformer_features, output_file)
    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
    


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    test_data = torch.load(os.path.join(data_dir, "test.pt"))
    return dict(train=train_data, test=test_data)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
