"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm
from datasets import load_dataset

IGNORE_INDEX = -1


def prepare(
    destination_path: Path = Path("data/lima"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    max_seq_length: int = 512,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    data_source: str = "GAIR/lima"
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    destination_path.mkdir(parents=True, exist_ok=True)

    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)
    lima_dataset = load_dataset(data_source)

    # Partition the dataset into train and test
    train_set, test_set = lima_dataset["train"], lima_dataset["test"]
    train_set, test_set = list(train_set), list(test_set)

    train_set = [sample for sample in train_set if sample["source"] != "multi_turn"]

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set)]
    torch.save(test_set, destination_path / "test.pt")


def prepare_sample(example: list, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
    """Processes a single sample.

    Currently not contains the multi turn conversations.
    """
    assert len(example["conversations"]) <= 2
    full_prompt = example["conversations"][0]

    TRAIN = (len(example["conversations"]) == 2)
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)

    if TRAIN:
        full_prompt_and_response = full_prompt + example["conversations"][1]
        encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

        # The labels are the full prompt with response, but with the prompt masked out
        labels = encoded_full_prompt_and_response.clone()
        if mask_inputs:
            labels[:len(encoded_full_prompt)] = IGNORE_INDEX

        return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}
    else:
        return {**example, "input_ids": encoded_full_prompt}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
