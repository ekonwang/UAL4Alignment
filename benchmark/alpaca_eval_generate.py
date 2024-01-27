import sys
import os
import json
import time
import warnings
import datetime
from pathlib import Path
from typing import Optional

import datasets
import lightning as L
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from scripts.prepare_llama2 import generate_prompt
# model generation utils
from utils import load_causal_model, model_generate

def main(
    lora_path: Path = None,
    pretrained_model_tag: str = 'mistral-7b',
    max_new_tokens: int = 1536,
    max_tokens: int = 512,
    top_k: int = 200,
    temperature: float = 0.8,
    shot_num: int = 0,
    output_file: str = None,
) -> None:
    
    assert Path(lora_path).is_file()
    lora_signature = f"{'/'.join(str(lora_path).rsplit('.', 1)[0].split('/')[-3:])}"
    pretrained_model_tag = str(lora_path).split('/')[-3] # mistral-7b or llama2-7b
    assert pretrained_model_tag in ['mistral-7b', 'llama2-7b']

    # output to on-disk file
    output_file = Path(f"out/benchmark/"\
                    f"alpaca_eval/"\
                    f"{lora_signature}"\
                    f".json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # if output_file.is_file():
    #     print(f'File {output_file} exists. Exiting..')
    #     exit(0)
    print(output_file)

    # model setup
    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)
    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    model, tokenizer = load_causal_model(pretrained_model_tag, lora_path, fabric)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    model.eval()
    model = fabric.setup(model)

    # data setup
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    eval_set = [e for e in eval_set]

    # generate
    outputs = []
    for eval_sample in tqdm(eval_set):
        instruction = eval_sample["instruction"]
        prompt = generate_prompt(instruction)
        output = model_generate(model, tokenizer, prompt, 
                                model_tag=pretrained_model_tag, 
                                max_tokens=max_tokens,
                                max_new_tokens=max_new_tokens,
                                top_k=top_k, temperature=temperature)
        
        eval_sample['output'] = output
        eval_sample['generator'] = lora_signature
        outputs.append(dict(
            **eval_sample,
        ))
        print(prompt, output)
        print('\n=============================\n')
  
        # save to disk
        with open(output_file, "w") as f:
            json.dump(outputs, f, indent=4)
        print(f"\nSaved to {output_file}")
    
    
if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
