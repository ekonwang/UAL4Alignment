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

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup
from scripts.prepare_lima import generate_prompt

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05

data_configs = {
    "ARC": ("ai2_arc", "ARC-Challenge", "test"),
    "MMLU": ("cais/mmlu", "all", "test"),
    "TruthfulQA": ("truthful_qa", "multiple_choice", "validation"),
}
choice_configs = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]

def main(
    lora_path: Path = Path("out/lora/lima/lit-llama-lora-finetuned.pth"),
    pretrained_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = None,
    max_new_tokens: int = 1024,
    top_k: int = 200,
    temperature: float = 0.8,
    data_dir: str = "ARC",
    shot_num: int = 0,
    output_file: str = None,
) -> None:
    
    assert lora_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()
    assert data_dir in data_configs.keys()
    assert shot_num <= 32

    if output_file is None:
        output_file = f"out/benchmark/multi-choices/lima_{data_dir}/{'-'.join(str(lora_path).split('/')[-2:]).rsplit('.', 1)[0]}_{datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')}.json"

    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA.from_name(name)

            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned lora weights
            model.load_state_dict(lora_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)
    tokenizer = Tokenizer(tokenizer_path)

    collected_responses = list()
    dataset = data_preprocess(data_dir)
    test_dataset, icl_dataset = dataset[:len(dataset)-shot_num], dataset[-shot_num:]

    for sample in tqdm(test_dataset):
        return_dict = generate_inputs(sample, icl_dataset)
        prompt = generate_prompt(return_dict['inputs'])

        # output = model_inference(model, tokenizer, prompt, max_new_tokens, top_k, temperature)
        # output = tokenizer.decode(output)
        # response = output.replace(prompt, "").strip()
        response = model_fast_inference(model, tokenizer, prompt, num_choices=return_dict['num_choices'])

        print(prompt)
        print(response)
        print("\n\n====================\n")

        collected_responses.append(dict(
            answer=sample['answer'],
            inputs=return_dict['inputs'],
            prompt=prompt,
            response=response,
        ))

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(collected_responses, f, indent=4)
        print(f"Saved to {output_file}", file=sys.stderr)

    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


def model_inference(model, tokenizer, prompt, max_new_tokens, top_k, temperature):
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    t0 = time.perf_counter()
    output = generate(
        model,
        idx=encoded,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id
    )
    model.reset_cache()
    t = time.perf_counter() - t0
    print(f"\n\nTime for inference: {t:.02f} sec total, {(len(output) - len(encoded)) / t:.02f} tokens/sec", file=sys.stderr)
    return output


def model_fast_inference(model, tokenizer, prompt, num_choices):
    # TODO: implement fast inference by looking into logit probability of choices such as 'A', 'B', 'C' and 'D'
    # inference code: https://github.com/hendrycks/test/blob/master/evaluate_flan.py#L72C9-L72C9
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device).unsqueeze(0)
    choices = choice_configs[:num_choices]
    choices_idx = torch.tensor([tokenizer.encode(c, bos=False)[0] for c in choices]).unsqueeze(0)

    logits = model(encoded) # size is (1, V)
    gathered = torch.gather(logits, 1, choices_idx)
    model_choice = choices[torch.argmax(gathered).item()]
    return model_choice


def wrap_input(sample_dict, include_answer=False):
    question_with_answer_temp = f"""
    # Question: {sample_dict['question']}
    
    # Choices: 
    """ 
    for idx, choice in zip(choice_configs, sample_dict['choices']):
        question_with_answer_temp += f"{idx}. {choice}\n"
    if include_answer:
        question_with_answer_temp += f"\n# Answer: {sample_dict['answer']}\n"
    return question_with_answer_temp


def generate_inputs(sample_dict, icl_dataset):
    # TODO: generate the input/instruction for the model

    prompt = """The following are multiple choice questions (with answers)
    """
    for icl_sample in icl_dataset:
        prompt += wrap_input(icl_sample, include_answer=True)
    prompt += wrap_input(sample_dict, include_answer=False)


def data_preprocess(data_dir):
    """ 1. load the data with `datasets` lib.
        2. process multi-choice data dictionary to be unified format, to fit into the fixed prompt.
    """
    data_path, subset, split = data_configs[data_dir]
    dataset = load_dataset(data_path, subset)[split]
    label_map = {v: c for v, c in zip(range(26), choice_configs)}

    processed = []
    for sample in dataset:
        if data_dir == "ARC":
            sample_dict = dict(
                question=sample["question"],
                choices=sample["choices"]["text"],
                answer=sample["answerKey"],
            )
        elif data_dir == "MMLU":
            sample_dict = dict(
                question=sample["question"],
                choices=sample["choices"]["text"],
                answer=sample["answerKey"],
            )
        elif data_dir == "TruthfulQA":
            sample_dict = dict(
                question=sample["question"],
                choices=sample["mc1_targets"]["choices"],
                answer='A',  # becuase the answer is always the first choice, please refer https://huggingface.co/datasets/truthful_qa/viewer/multiple_choice/validation?p=8&row=800
            )
        else:
            raise ValueError(f"Unknown data_dir: {data_dir}")
        processed.append(sample_dict)

    return processed
        

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
