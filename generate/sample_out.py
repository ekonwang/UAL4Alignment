import sys
from pathlib import Path
import time
import warnings
import json

import openai
import torch
from datasets import load_dataset
import lightning as L

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from benchmark.utils import load_causal_model, model_generate
from scripts.prepare_llama2 import generate_prompt

def main(
    prompt1: str = "Please tell me how to become successful.",
    *,
    lora_path: Path = None,
    pretrained_model_tag: str = 'mistral-7b',
    data_path: str = 'GAIR/lima',
    max_new_tokens: int = 1568,
    top_k: int = 200,
    temperature: float = 0.8,
    num_samples: int = 3,
):
    if lora_path is not None:
        pretrained_model_tag = str(lora_path).split('/')[-3]
    assert pretrained_model_tag in ['llama2-7b', 'llama2-13b', 'mistral-7b']

    model_split = str(lora_path).split('/')[-5]
    lora_signature = '/'.join(str(lora_path).rsplit('.', 1)[0].split('/')[-3:]) # base model / sft model signature / model steps
    dataset = load_data(data_path)
    out_path = Path(f'out/text/{model_split}/{lora_signature}.json')
    out_path.parent.mkdir(exist_ok=True, parents=True)

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)
    print("Loading model ...", file=sys.stderr)
    t0 = time.perf_counter()
    model, tokenizer = load_causal_model(pretrained_model_tag, lora_path, fabric)
    print(f"Time to load model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    model.eval()
    model = fabric.setup(model)

    out_list_ = []
    for sample in dataset:
        instruction = sample['instruction']
        prompt = generate_prompt(instruction)


        print('>>>>>')
        print(prompt)
        responses = []
        resp_top = None
        for i in range(num_samples):
            resp = model_generate(
                model, tokenizer,
                prompt, model_tag=pretrained_model_tag,
                max_tokens=512, max_new_tokens=max_new_tokens,
                top_k=top_k, temperature=temperature,
            )
            responses.append(resp)
            resp_top = resp if (resp_top is None or len(resp) > len(resp_top)) else resp_top
        # select the longest response and print
        print(resp_top)

        out_list_.append(dict(
            **sample,
            responses=responses
        ))
        with open(out_path, 'w') as f:
            json.dump(out_list_, f, indent=4)
    

def load_data(data_path):
    if data_path == 'GAIR/lima':
        raw_data = load_dataset(data_path, split='test')
    else:
        raise NotImplementedError
    out_list_ = []
    for sample in raw_data:
        out_ = dict()
        if data_path == 'GAIR/lima':
            instruction = sample['conversations'][0]
        out_['instruction'] = instruction
        out_list_.append(out_)
    return out_list_

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore", 
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    CLI(main)
