"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json

import numpy as np
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm
from datasets import load_dataset

IGNORE_INDEX = -1


def prepare(
    destination_path: Path = Path("data/lima"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    max_seq_length: int = 512,  # 2048 for the setting in the paper (https://arxiv.org/pdf/2305.11206.pdf)
    seed: int = 42,
    mask_inputs: bool = False,  # not as in alpaca-lora, we have multi-turn dialogue
    data_source: str = "GAIR/lima",
    score_path: str = None
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

    # train_set = [sample for sample in train_set if sample["source"] != "multi_turn"]
    # train_set = [sample for sample in train_set]

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)]

    if score_path is not None:
        with open(score_path) as f:
            scores = json.load(f)
            assert len(scores) == len(train_set)
        scores = make_score_dist(scores, target_mean=0.1)
        train_set = [dict(sample, smooth_value=score) for sample, score in zip(train_set, scores)]

    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set)]
    torch.save(test_set, destination_path / "test.pt")


def make_score_dist(raw_scores, target_mean=0.1, max_values=0.99):
    assert target_mean <= max_values

    scores = np.array(raw_scores)
    while(abs(np.mean(scores) - target_mean) >= 0.001):
        scores = scores / np.mean(scores) * target_mean
        scores = np.clip(scores, scores.min(), max_values)
        print(np.mean(scores))    
    
    return scores.tolist()


def prepare_sample(example: list, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True) -> dict:
    """Processes a single sample.

    Currently not contains the multi turn conversations.
    """
    assert 1 <= len(example["conversations"]) <= 2 or (example["source"] == "multi_turn")

    instructions = example["conversations"][::2]
    responses = example["conversations"][1::2]
    has_sys_prompt = False

    if len(responses) == 0: # for the test set
        full_prompt = generate_prompt(instructions[0], sys=has_sys_prompt)
        encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
        return {**example, "input_ids": encoded_full_prompt}
    else:
        label_list = []
        dialogue_list = []

        for instr, resp in zip(instructions, responses): # iteratively concatenate the dialogue turns
            if len(dialogue_list) == 0:
                full_prompt = generate_prompt(instr, sys=has_sys_prompt)
            else:
                full_prompt = generate_prompt(instr)
            full_prompt_and_response = full_prompt + resp
            encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
            encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

            label = encoded_full_prompt_and_response.clone()
            if mask_inputs:
                label[:len(encoded_full_prompt)] = IGNORE_INDEX
            dialogue_list.append(encoded_full_prompt_and_response)
            label_list.append(label)
            # TODO:  1) to check if ignore_index is set correctly 2) to check if the prefix is correctly tokenized
        input_ids = torch.cat(dialogue_list, dim=0)
        labels = torch.cat(label_list, dim=0)
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

        return {**example, "input_ids": input_ids, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def generate_prompt(instruction, sys: bool = False) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    # vicuna style: https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
    SYS_PROMPT = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
"""
    DIALOGUE_PROMPT = f"""
### User: {instruction}

### Assistant: """
    
    if sys:
        return f"""{SYS_PROMPT}{DIALOGUE_PROMPT}"""
    else:
        return f"""{DIALOGUE_PROMPT}"""
    

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
