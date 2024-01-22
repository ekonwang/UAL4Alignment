import sys
import os
import json
import time
import warnings
import datetime
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from datasets import load_dataset
from peft import get_peft_model
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from transformers import AutoTokenizer, AutoModelForCausalLM

from openllm_leaderboard import (choice_configs, data_configs, lora_config,
                                 data_preprocess, model_best_of_n_inference, generate_inputs)
from utils import load_lora_ckpt_from_disk_to_hf_model

__HF_MODEL="mistralai/Mistral-7B-v0.1"

def main(
    lora_path: Path = None,
    max_tokens: int = 512,
    top_k: int = 200,
    temperature: float = 0.8,
    data_dir: str = "ARC",
    shot_num: int = 0,
    output_file: str = None,
    best_of: int = 4,
) -> None:
    
    assert data_dir in data_configs.keys()
    assert shot_num <= 32

    pretrained_path: str = __HF_MODEL
    lora_signature = '/'.join(str(lora_path).rsplit('.', 1)[0].split('/')[-3:]) # base model / sft model signature / model steps
    output_file = Path(f"out/benchmark/"\
                    f"best-of-n/"\
                    f"{data_dir}/"\
                    f"{shot_num}-shot/"\
                    f"{lora_signature}"\
                    f".json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.is_file():
        exit(0)
    print(output_file)

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    # model loading
    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(pretrained_path).half()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model = load_lora_ckpt_from_disk_to_hf_model(lora_path, model, lora_config=lora_config)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    # evaluation mode
    model.eval()
    model = fabric.setup(model)

    collected_responses = list()
    dataset = data_preprocess(data_dir)
    test_dataset, icl_dataset = dataset, []

    if shot_num > 0:
        test_dataset, icl_dataset = dataset[:len(dataset)-shot_num], dataset[-shot_num:]

    acc_cnt = 0
    tot_cnt = 0 
    inference_config = {
        "data_dir": data_dir,
        "shot_num": shot_num,
        "best_of": best_of,
    }
    for sample in tqdm(test_dataset):
        return_dict = generate_inputs(sample, icl_dataset, config=inference_config)
        prompt = return_dict['inputs']
        # Mistral loaded as hf format model
        response = model_best_of_n_inference(model, tokenizer, prompt, 
                                             max_tokens=max_tokens, 
                                             temperature=temperature, 
                                             top_k=top_k, n=best_of, 
                                             _hf_model=True)

        acc_cnt += 1 if sample['answer'] in response else 0
        tot_cnt += 1

        print(prompt, response)
        print("\n\n========== {}/{} ({:.2f} %) ==========\n".format(acc_cnt, tot_cnt, acc_cnt / tot_cnt * 100))

        collected_responses.append(dict(
            answer=sample['answer'],
            inputs=return_dict['inputs'],
            prompt=prompt,
            response=response,
        ))
        
    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
    
    collected_responses = [dict(acc=100*acc_cnt/tot_cnt)] + collected_responses

    with open(output_file, "w") as f:
        json.dump(collected_responses, f, indent=4)
    print(f"Saved to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
