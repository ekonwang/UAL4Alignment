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
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup
from scripts.prepare_lima import generate_prompt
from utils import load_causal_model

lora_r = 8
lora_alpha = 16
lora_dropout = 0.0


def main(
    lora_path: Path = None,
    model_tag: str = 'llama2-7b',
    data_dir: str = "./data/lima",
    output_file: Path = Path("./out/pca_analysis/llama2-7b-lima-test-features.pt"),
    data_split: str = "test",
) -> None:
    

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    model, tokenizer = load_causal_model(model_tag, lora_path, fabric)
    model.eval()
    model = fabric.setup(model)

    collected_transformer_features = list()
    encoded = None
    test_dataset = load_datasets(data_dir)[data_split]

    def transformer_hook(model, input, output):
        # the output is the results of the last RMS norm layer, size is (B, T, embed_size)
        embed_size = output.size(-1)
        features = output.clone().cpu().detach()[...,:-1,:].view(-1, embed_size)
        labels = encoded[...,1:].view(-1).tolist()

        assert len(labels) == features.size(0)
        
        for label, feature in zip(labels, features):
            collected_transformer_features.append((label, feature))
        print(len(labels))

    if model_tag == 'llama2-7b':
        hook = model.transformer.ln_f.register_forward_hook(transformer_hook)
    elif model_tag == 'mistral-7b':
        pass
    else:
        raise NotImplementedError(f'Unsupported model_tag: {model_tag}')

    for sample in tqdm(test_dataset):
        encoded = sample['input_ids'].view(1, -1).to(fabric.device)
        logits = model(encoded)
    
    hook.remove()
    torch.save(collected_transformer_features, output_file)
    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


def load_datasets(data_dir, model_tag="llama2-7b"):
    if model_tag == 'llama2-7b':
        train_data = torch.load(os.path.join(data_dir, "train.pt"))
        test_data = torch.load(os.path.join(data_dir, "test.pt"))
    elif model_tag == 'mistral-7b':
        train_data = torch.load(os.path.join(data_dir, "train_Mistral-7B-v0.1.pt"))
        test_data = torch.load(os.path.join(data_dir, "test_Mistral-7B-v0.1.pt"))
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
