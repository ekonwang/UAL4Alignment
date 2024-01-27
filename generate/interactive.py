import sys
from typing import Optional
from pathlib import Path
import time
import warnings

import openai
import torch
from peft import get_peft_model, PeftModel, PeftMixedModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import lightning as L

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup
from lit_llama import Tokenizer, LLaMA
from benchmark.utils import load_causal_model, model_generate
from scripts.prepare_llama2 import generate_prompt

def main(
    prompt1: str = "Please tell me how to become successful.",
    *,
    lora_path: Path = None,
    pretrained_model_tag = None,
    num_samples: int = 1,
    max_new_tokens: int = 1568,
    top_k: int = 200,
    temperature: float = 0.8,
):
    if lora_path is not None:
        pretrained_model_tag = str(lora_path).split('/')[-3]
    assert pretrained_model_tag in ['llama2-7b', 'llama2-13b', 'mistral-7b']

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)
    print("Loading model ...", file=sys.stderr)
    t0 = time.perf_counter()
    model, tokenizer = load_causal_model(pretrained_model_tag, lora_path, fabric)
    print(f"Time to load model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    model.eval()
    model = fabric.setup(model)

    while 1:
        instruction = input("Please input your prompt: ")
        if len(instruction) <= 5:
            instruction = prompt1
        prompt = generate_prompt(instruction)

        print(prompt)
        response = model_generate(
            model, tokenizer,
            prompt, model_tag=pretrained_model_tag,
            max_tokens=512, max_new_tokens=max_new_tokens,
            top_k=top_k, temperature=temperature,
            llama_stream=True
        )
        print('\n\n>>>>>> context clear!!! =====')
        # print(response)


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
