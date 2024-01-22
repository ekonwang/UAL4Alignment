import openai
import torch
from peft import get_peft_model, PeftModel, PeftMixedModel


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
