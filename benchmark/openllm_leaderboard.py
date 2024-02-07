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
from peft import LoraConfig
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# load dataset offline for robustness
os.environ["HF_DATASETS_OFFLINE"] = "1"

from generate import generate
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup
from scripts.prepare_lima import generate_prompt
from utils import lora_alpha, lora_dropout, lora_r, lora_config, load_causal_model

data_configs = {
    "ARC": ("ai2_arc", "ARC-Challenge", "validation"),
    "MMLU": ("cais/mmlu", "all", "validation"),  # https://huggingface.co/datasets/cais/mmlu
    "TruthfulQA": ("truthful_qa", "multiple_choice", "validation"),
    "HellaSwag": ("Rowan/hellaswag", None, "validation"),  # https://huggingface.co/datasets/Rowan/hellaswag
}
choice_configs = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]

def main(
    lora_path: Path = None,
    pretrained_tag: str = 'llama2-7b',
    quantize: Optional[str] = None,
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

    model_split = str(lora_path).split('/')[-5]
    lora_signature = '/'.join(str(lora_path).rsplit('.', 1)[0].split('/')[-3:]) # base model / sft model signature / model steps
    output_file = Path(f"out/benchmark/"\
                    f"best-of-n/{model_split}/"\
                    f"{data_dir}/"\
                    f"{shot_num}-shot/"\
                    f"{lora_signature}"\
                    f".json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.is_file():
        print(f"File {output_file} already exists, skip.", file=sys.stderr)
        exit(0)
    print(output_file)

    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")
    dataset = data_preprocess(data_dir)

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    model, tokenizer = load_causal_model(pretrained_model_tag=pretrained_tag, lora_path=lora_path, fabric=fabric)
    tokenizer = tokenizer.processor
    # model = load_causal_model(pretrained_path, lora_path, fabric)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    collected_responses = list()
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
    if 'llama2' in pretrained_tag:
        hf_signature = False 
    elif 'mistral' in pretrained_tag:
        hf_signature = True
    else:
        raise NotImplementedError()
    for sample in tqdm(test_dataset):
        return_dict = generate_inputs(sample, icl_dataset, config=inference_config)
        # prompt = generate_prompt(return_dict['inputs'])
        prompt = return_dict['inputs']
        # output = model_inference(model, tokenizer, prompt, max_new_tokens, top_k, temperature)
        # response = model_fast_inference(model, tokenizer, prompt, num_choices=return_dict['num_choices'], max_tokens=max_tokens)
        response = model_best_of_n_inference(model, tokenizer, prompt, 
                                             max_tokens=max_tokens, 
                                             temperature=temperature, 
                                             top_k=top_k, n=best_of, _hf_model=hf_signature)

        # acc_cnt += 1 if response == sample['answer'] else 0
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
    output = tokenizer.decode(output)
    response = output.replace(prompt, "").strip()
    return response


def model_fast_inference(model, tokenizer, prompt, num_choices, max_tokens):
    # TODO: implement fast inference by looking into logit probability of choices such as 'A', 'B', 'C' and 'D'
    # inference code: https://github.com/hendrycks/test/blob/master/evaluate_flan.py#L72C9-L72C9
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)[-max_tokens:].unsqueeze(0)
    choices = choice_configs[:num_choices]
    choices_idx = torch.tensor([tokenizer.encode(c, bos=False)[0] for c in choices], device=model.device, dtype=torch.long)
    
    try:
        logits = model(encoded)[0, -1] # size is (V)
    except Exception as e:
        print(encoded.shape)
        print(e)
        import pdb; pdb.set_trace()

    gathered = torch.gather(logits, 0, choices_idx)
    model_choice = choices[torch.argmax(gathered).item()]
    return model_choice


def model_best_of_n_inference(model, tokenizer, prompt, max_tokens, temperature, top_k, n=32, _hf_model=False):
    """
    The function is equivalent to model generate n times with temperature and top_k.
    And the model is asked to output in following format:

    [Choice]. [Rationale].
     """
    # TODO: model must output a choice first, then generate the rationales.
    # so we do not bother generating all the tokens, just care about the first one.
    if not _hf_model:
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
        encoded = encoded[-max_tokens:].unsqueeze(0)
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = encoded['input_ids'] if not isinstance(encoded, torch.Tensor) else encoded
        encoded = encoded[..., -max_tokens:].to(model.device)
    # it is up to the model to generate any choice, even out of the candidate choices (e.g., 'E', 'F', 'G', 'H')
    choices = choice_configs[:]
    # choices_idx = torch.tensor([tokenizer.encode(c, bos=False)[0] for c in choices], device=model.device, dtype=torch.long)

    # size is (V)
    outputs = model(encoded) if not _hf_model else model(encoded).logits
    logits = outputs[0, -1] / temperature
    # scatter -float("Inf") to the elements in the logits if not in topk
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

    # scatter -float("Inf") to the elements in the logits if the index not in choices_idx
    # logits = torch.where(torch.isin(torch.arange(logits.size(-1), device=logits.device), choices_idx), logits, -float("Inf"))
    probs = torch.nn.functional.softmax(logits, dim=-1)

    idxs = []
    for i in range(n):
        idxs.append(int(torch.multinomial(probs, num_samples=1)[0].to(dtype=torch.long)))

    if not _hf_model:
        decoded_answers = [tokenizer.processor.decode(idx) for idx in idxs]
    else:
        decoded_answers = [tokenizer.decode(idx) for idx in idxs]
    return decoded_answers


def wrap_input(sample_dict, config, include_answer=False):
    if config['data_dir'] == "HellaSwag":
        question_with_answer_temp = f"""
# Context:
{sample_dict['question']}

# Choices:
""" 
    else:        
        question_with_answer_temp = f"""
# Question:
{sample_dict['question']}

# Choices:
""" 
    for idx, choice in zip(choice_configs, sample_dict['choices']):
        question_with_answer_temp += f"{idx}. {choice}\n"
    question_with_answer_temp += f"\n# Answer:\n"
    if include_answer:
        question_with_answer_temp += f"{sample_dict['answer']}\n"
    return question_with_answer_temp


def generate_inputs(sample_dict, icl_dataset, config: dict):
    # generate the input/instruction for the model

    if config['data_dir'] == "HellaSwag":
        prompt = """Which of the above options is the most sensible continuation of the given context? Choose the correct letter (A, B, C, or D).
"""
    else:
        prompt = """The following are multiple choice questions (with answers), which is the correct answer? Choose the correct letter.
"""
    for icl_sample in icl_dataset:
        prompt += wrap_input(icl_sample, config, include_answer=True)
    prompt += wrap_input(sample_dict, config, include_answer=False)

    return dict(
        inputs=prompt,
        num_choices=len(sample_dict['choices']),
    )


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
                choices=sample["choices"],
                answer=label_map[sample["answer"]],
            )
        elif data_dir == "TruthfulQA":
            sample_dict = dict(
                question=sample["question"],
                choices=sample["mc1_targets"]["choices"],
                answer='A',  # becuase the answer is always the first choice, please refer https://huggingface.co/datasets/truthful_qa/viewer/multiple_choice/validation?p=8&row=800
            )
        elif data_dir == "HellaSwag":
            sample_dict = dict(
                question=sample["ctx"],
                choices=sample["endings"],
                answer=label_map[int(sample["label"])],
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
