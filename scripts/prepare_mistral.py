"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
import os
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import json

import numpy as np
from torch.utils.data import random_split
from transformers import AutoTokenizer
from tqdm import tqdm

from prepare_llama2 import IGNORE_INDEX, generate_prompt, prepare_sample


def prepare( 
    tokenizer_path: Path = Path("mistralai/Mistral-7B-v0.1"),
    max_seq_length: int = 1024,  # half of 2048 (the setting in the paper) (https://arxiv.org/pdf/2305.11206.pdf)
    seed: int = 42,
    mask_inputs: bool = False,  # not as in alpaca-lora, we have multi-turn dialogue
    data_source: str = "data/meta-math-6k/data.json",
    score_path: str = None,
    smooth_value: float = 0.1,
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    destination_path = Path(data_source).parent
    if score_path is not None:
        destination_path = destination_path / f"ls-{smooth_value}"
    destination_path.mkdir(parents=True, exist_ok=True)
    print(destination_path)

    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    with open(data_source, 'r') as f:
        lima_dataset = json.load(f)
    test_source = data_source.replace('data.json', 'data_test.json')
    if os.path.exists(test_source):
        with open(test_source, 'r') as f:
            lima_dataset_test = json.load(f)
    else:
        lima_dataset_test = None

    # Partition the dataset into train and test
    train_set = lima_dataset
    print(f"train has {len(train_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, tokenize, max_seq_length, mask_inputs) for sample in tqdm(train_set)]
    if lima_dataset_test is not None:
        test_set = [prepare_sample(sample, tokenizer, tokenize, max_seq_length, mask_inputs) for sample in tqdm(lima_dataset_test)]
        torch.save(test_set, destination_path / 'test_Mistral-7B-v0.1.pt')
    print(tokenizer.decode(train_set[-2]['input_ids']))
    torch.save(train_set, destination_path / "train_Mistral-7B-v0.1.pt")


def tokenize(tokenizer: AutoTokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    encoded = tokenizer.encode(string, max_length=max_length, truncation=True) + ([tokenizer.eos_token_id] if eos else [])
    return torch.tensor(encoded, dtype=torch.long)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
