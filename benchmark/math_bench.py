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
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup
from scripts.prepare_llama2 import generate_prompt
from utils import load_lora_ckpt_from_disk_to_hf_model


lora_r = 8
lora_alpha = 16
lora_dropout = 0.0

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

def main(
    lora_path: Path = None,
    pretrained_model_tag: str = 'llama2-7b',
    max_new_tokens: int = 256,
    max_tokens: int = 512,
    top_k: int = 200,
    temperature: float = 0.8,
    data_dir: str = "gsm8k",
    shot_num: int = 0,
    output_file: str = None,
    best_of: int = 1,
) -> None:
    assert shot_num <= 32

    lora_signature = f"{'/'.join(str(lora_path).rsplit('.', 1)[0].split('/')[-2:])}" if lora_path is not None else pretrained_model_tag
    output_file = Path(f"out/benchmark/"\
                    f"math/"\
                    f"{data_dir}/"\
                    f"{shot_num}-shot/"\
                    f"{lora_signature}_{data_dir}"\
                    f".json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.is_file():
        exit(0)
    print(output_file)

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    model, tokenizer = load_causal_model(pretrained_model_tag, lora_path, fabric)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    collected_responses = list()
    dataset = data_preprocess(data_dir)
    test_dataset, icl_dataset = dataset, []

    acc_cnt = 0
    tot_cnt = 0 
    inference_config = {
        "data_dir": data_dir,
        "shot_num": shot_num,
        "best_of": best_of,
    }
    if shot_num > 0:
        test_dataset, icl_dataset = dataset[:len(dataset)-shot_num], dataset[-shot_num:]
    for sample in tqdm(test_dataset):
        return_dict = generate_inputs(sample, icl_dataset, config=inference_config)
        prompt = return_dict['inputs']
        
        responses = []
        for i in range(best_of):
            resp = model_generate(model, tokenizer, prompt, pretrained_model_tag, 
                                  max_tokens=max_tokens, max_new_tokens=max_new_tokens, 
                                  top_k=top_k, temperature=temperature)
            responses.append(resp)
        
        response = responses[0]
        for resp in responses:
            if sample['answer'] in resp:
                response = resp
                acc_cnt += 1
                break
        
        print(prompt, response)
        tot_cnt += 1
        print("Ground truth: ", sample['answer'])
        print("\n")
        print("========== {}/{} ({:.2f} %) ==========".format(acc_cnt, tot_cnt, acc_cnt / tot_cnt * 100))
        print("\n")

        collected_responses.append(dict(
            answer=sample['answer'],
            inputs=return_dict['inputs'],
            prompt=prompt,
            response=responses,
        ))
        

    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
    
    collected_responses = [dict(acc=100*acc_cnt/tot_cnt)] + collected_responses

    with open(output_file, "w") as f:
        json.dump(collected_responses, f, indent=4)
    print(f"Saved to {output_file}", file=sys.stderr)


def generate_inside_parts(sample, include_response=False):
    if not include_response:
        temp = '''
# Question:
```
{question}
```
'''
    else:
        temp = '''
# Question:
```
{question}
```
# Answer:
```
{response}
```
'''
    return temp.format_map(sample)


def generate_inputs(sample, icl_dataset, config):
    instruction = '''Here are some questions and answers about math:

'''
    for example in icl_dataset:
        instruction += generate_inside_parts(example, include_response=True)
    instruction += generate_inside_parts(sample, include_response=False)
    prompt = generate_prompt(instruction)
    return dict(inputs=prompt)


def model_generate(model, tokenizer, prompt, model_tag,
                   max_tokens, max_new_tokens, top_k, temperature):
    encoded = tokenizer.encode(prompt).to(model.device)[-max_tokens:]

    if model_tag == 'llama2-7b':
        output = generate(
            model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.processor.eos_id
        )
        model.reset_cache()
    elif model_tag == 'mistral-7b':
        output = model.generate(
            input_ids=encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.processor.eos_token_id,
            early_stopping=True,
        )

    output = tokenizer.decode(output)
    response = output.replace(prompt, "").strip()
    return response


def data_preprocess(data_dir):
    if data_dir == 'gsm8k':
        dataset = load_dataset("gsm8k", "main", split="test")
    elif data_dir == 'meta-math':
        dataset = load_dataset("meta-math/MetaMathQA")['train']
        # select the last 1000 samples as test set
        # NOTE: the set of samples not appear in the training set
        dataset = [d for d in dataset][-1000:]

    processed = []
    for sample in dataset:
        if data_dir == 'gsm8k':
            processed.append(dict(
                question = sample['question'],
                answer = sample['answer'].rsplit('####', 1)[1].strip(),
                response = sample['answer'].rsplit('####', 1)[0].strip()
            ))
        elif data_dir == 'meta-math':
            processed.append(dict(
                question = sample['query'],
                # detect the last word as the ground truth
                answer = sample['response'].strip(' .').split()[-1],
                response = sample['response'].rsplit('####', 1)[0].strip()
            ))
        else:
            print(data_dir)
            raise NotImplementedError
    return processed


class GeneralTokenizer:
    def __init__(self, tokenizer, model_tag):
        self.processor = tokenizer 
        self.model_tag = model_tag 
    
    def encode(self, string):
        if self.model_tag == 'llama2-7b':
            return self.processor.encode(string, bos=True, eos=False)
        elif self.model_tag == 'mistral-7b':
            return self.processor.encode(string, add_special_tokens=True, return_tensors='pt')

    def decode(self, tokens):
        if self.model_tag == 'llama2-7b':
            return self.processor.decode(tokens)
        elif self.model_tag == 'mistral-7b':
            tokens = tokens.squeeze()
            return self.processor.decode(tokens)


def load_causal_model(pretrained_model_tag, lora_path, fabric):
    # model tag should be inside the lora path 
    if lora_path is not None:
        assert pretrained_model_tag in str(lora_path)

    if pretrained_model_tag == 'llama2-7b':
        pretrained_path = 'checkpoints/lit-llama/7B/lit-llama.pth'
        tokenizer = Tokenizer('checkpoints/lit-llama/tokenizer.model')

        with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
            name = llama_model_lookup(pretrained_checkpoint)

            with fabric.init_module(empty_init=True), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
                model = LLaMA.from_name(name)
                # 1. Load the pretrained weights
                model.load_state_dict(pretrained_checkpoint, strict=False)
                # 2. Load the fine-tuned lora weights
                if lora_checkpoint is not None:
                    model.load_state_dict(lora_checkpoint, strict=False)
    
    elif pretrained_model_tag == 'mistral-7b':
        model_name_or_path = "mistralai/Mistral-7B-v0.1"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # make lora model
        if lora_path is not None:
            model = load_lora_ckpt_from_disk_to_hf_model(lora_path, model, lora_config=lora_config)
                
    return model, GeneralTokenizer(tokenizer, pretrained_model_tag)
        

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
