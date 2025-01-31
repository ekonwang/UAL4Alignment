"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import json

import numpy as np
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm

IGNORE_INDEX = -1


def prepare(
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
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
    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)
    with open(data_source, 'r') as f:
        lima_dataset = json.load(f)

    destination_path = Path(data_source).parent
    if score_path is not None:
        destination_path = destination_path / f"ls-{smooth_value}"
    destination_path.mkdir(parents=True, exist_ok=True)
    print(destination_path)

    # Partition the dataset into train and test
    train_set = lima_dataset
    print(f"train has {len(train_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, tokenize, max_seq_length, mask_inputs) for sample in tqdm(train_set)]
    print(tokenizer.decode(train_set[-1]['input_ids']))
    torch.save(train_set, destination_path / "train.pt")
    with open(destination_path / "train.json", "w") as f:
        json.dump([sample['decoded_inputs'] for sample in train_set], f, indent=4)


def prepare_sample(example: list, tokenizer: Tokenizer, tokenize_fn, max_length: int, mask_inputs: bool = True) -> dict:
    """Processes a single sample.

    Currently not contains the multi turn conversations.
    """
    instructions = [converse['value'] for converse in example["conversations"][::2]]
    responses = [converse['value'] for converse in example["conversations"][1::2]]
    has_sys_prompt = False
    accumulated_length = 0

    if len(responses) == 0: # for the test set
        full_prompt = generate_prompt(instructions[0], sys=has_sys_prompt)
        encoded_full_prompt = tokenize_fn(tokenizer, full_prompt, max_length=max_length, eos=False)
        return {**example, "input_ids": encoded_full_prompt}
    else:
        label_list = []
        dialogue_list = []

        for i, (instr, resp) in enumerate(zip(instructions, responses)): # iteratively concatenate the dialogue turns
            if len(dialogue_list) == 0:
                full_prompt = generate_prompt(instr, sys=has_sys_prompt)
            else:
                full_prompt = generate_prompt(instr)

            full_prompt_and_response = full_prompt + resp
            encoded_full_prompt = tokenize_fn(tokenizer, full_prompt, max_length=max_length, eos=False)
            encoded_full_prompt_and_response = tokenize_fn(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

            label = encoded_full_prompt_and_response.clone()
            if mask_inputs:
                label[:len(encoded_full_prompt)] = IGNORE_INDEX
            dialogue_list.append(encoded_full_prompt_and_response)
            label_list.append(label)
            accumulated_length += len(encoded_full_prompt_and_response)

            if accumulated_length >= max_length:
                break
            # TODO:  1) to check if ignore_index is set correctly 2) to check if the prefix is correctly tokenized

        input_ids = torch.cat(dialogue_list, dim=0)
        labels = torch.cat(label_list, dim=0)
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

        return {**example, "input_ids": input_ids, "labels": labels, "decoded_inputs": tokenizer.decode(input_ids)}


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
