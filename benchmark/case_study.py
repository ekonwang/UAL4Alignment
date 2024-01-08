import sys
import os
import json
import time
import warnings
from pathlib import Path

import lightning as L
import torch
from datasets import load_dataset
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup
from scripts.prepare_lima import generate_prompt

lora_r = 8
lora_alpha = 16
lora_dropout = 0.0


def main(
    loras_path: Path = Path("out/lora/lima"),
    pretrained_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    max_new_tokens: int = 1024,
    top_k: int = 200,
    temperature: float = 0.8,
    data_dir: Path = Path("out/benchmark/case_study/elected_samples.json"),
    output_file: str = None,
) -> None:
    
    assert loras_path.is_dir()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()
    assert str(loras_path).split('/')[-1]

    if output_file is None:
        output_file = f"out/benchmark/case_study/lima/{str(loras_path).split('/')[-1]}.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    with torch.load(pretrained_path) as pretrained_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA.from_name(name)

            # Load the pretrained weights
            import pdb; pdb.set_trace()
            # TODO: save a checkpoint matrix
            model.load_state_dict(pretrained_checkpoint, strict=False)

    print(f"Time to load pretrained model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model = fabric.setup(model)
    tokenizer = Tokenizer(tokenizer_path)
    loras = [lora_ckpt for lora_ckpt in sorted(os.listdir(loras_path)) if 'iter' in lora_ckpt]

    collected_responses = list()
    with open(data_dir, 'r') as f:
        prompt_sets = json.load(f)

    for pset in prompt_sets:
        inputs = prompt_sets[pset]

        for sample in tqdm(inputs):
            sample['set'] = pset
            collected_responses.append(
                dict(sample=list())
            )
            input = sample['conversations'][0]
            prompt = generate_prompt(input)
            encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

            for lora_ckpt in loras:
                # Split LoRA weights from pretrained weights
                model.train()
                with lora(lora_r, lora_alpha, lora_dropout), lazy_load(os.path.join(loras_path, lora_ckpt)) as lora_checkpoint:
                    # Load the LoRA weights
                    model.load_state_dict(lora_checkpoint, strict=False)
                # Merge LoRA weights into pretrained weights
                model.eval()

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
                print(os.path.join(loras_path, lora_ckpt))
                print(prompt)
                print(response)

                collected_responses[sample].append(
                    dict(
                        lora_ckpt=lora_ckpt,
                        response=response,
                    )
                )

                with open(output_file, "w") as f:
                    json.dump(collected_responses, f, indent=4)
                print(f"Saved to {output_file}", file=sys.stderr)

    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
