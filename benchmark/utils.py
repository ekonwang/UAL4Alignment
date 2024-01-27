import sys

import openai
import torch
from pathlib import Path
from peft import get_peft_model, PeftModel, PeftMixedModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup
from lit_llama import Tokenizer, LLaMA


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


def get_gpt_response(params, messages=None, temperature=None):
    resp = openai.ChatCompletion.create(
        model=params.model_name,
        messages=messages,
        temperature=params.temperature if temperature is None else temperature,
        max_tokens=params.max_tokens,
        top_p=params.top_p,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return resp["choices"][0]["message"]["content"]

def load_lora_ckpt_from_disk_to_hf_model(lora_path, model, lora_config=None):
    if not isinstance(model, PeftModel) and not isinstance(model, PeftMixedModel):
        model = get_peft_model(model, lora_config)

    lora_ckpt = torch.load(lora_path)
    model.load_state_dict(lora_ckpt, strict=False)
    # TODO: check sanity 
    # merge LoRA parameters into model to accelerate inference.
    model.merge_adapter()
    # NOTES: use `unmerge_adapter` to separate LoRA parameters from model.
    return model


class GeneralTokenizer:
    def __init__(self, tokenizer, model_tag):
        self.processor = tokenizer 
        self.model_tag = model_tag 
    
    def encode(self, string):
        if 'llama2' in self.model_tag:
            return self.processor.encode(string, bos=True, eos=False)
        elif self.model_tag == 'mistral-7b':
            return self.processor.encode(string, add_special_tokens=True, return_tensors='pt')

    def decode(self, tokens):
        if 'llama2' in self.model_tag:
            return self.processor.decode(tokens)
        elif self.model_tag == 'mistral-7b':
            tokens = tokens.squeeze()
            return self.processor.decode(tokens)


def model_generate(model, tokenizer, prompt, model_tag,
                   max_tokens, max_new_tokens, top_k, temperature, llama_stream=False):
    encoded = tokenizer.encode(prompt).to(model.device)[-max_tokens:]

    if 'llama2' in model_tag:
        output = generate(
            model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.processor.eos_id,
            tokenizer=tokenizer,
            stream=llama_stream
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


def load_causal_model(pretrained_model_tag, lora_path, fabric):
    # model tag should be inside the lora path 
    if lora_path is not None:
        assert pretrained_model_tag in str(lora_path)

    if 'llama2' in pretrained_model_tag:
        if pretrained_model_tag == 'llama2-7b':
            pretrained_path = 'checkpoints/lit-llama/7B/lit-llama.pth'
        if pretrained_model_tag == 'llama2-13b':
            pretrained_path = 'checkpoints/lit-llama/13B/lit-llama.pth'
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
