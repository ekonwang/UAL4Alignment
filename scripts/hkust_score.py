import torch 

from pathlib import Path
from model_utils import Llama_Scorer
from tqdm import tqdm
import json

def main(
    data_dir: str = None,
    output_dir: str = None,
):
    dataset = torch.load(data_dir)

    if output_dir is None:
        complexity_path = Path(data_dir).parent / "complexity_score.json"
        quality_path = Path(data_dir).parent / "quality_score.json"
        output_dir = Path(data_dir).parent / "scored.json"
    
    # complexity_score__ = infer_complexity(dataset)
    # with open(complexity_path, 'w') as f:
    #     json.dump(complexity_score__, f, indent=4)

    quality_score__ = infer_quality(dataset)
    with open(quality_path, 'w') as f:
        json.dump(quality_score__, f, indent=4)


def infer_complexity(dataset):
    model_name = "hkust-nlp/deita-complexity-scorer"
    scorer = Llama_Scorer(model_name_or_path=model_name, is_vllm=False)

    results = []
    for sample in tqdm(dataset):
        conversations = sample['conversations']
        instructions = [convers['value'] for convers in conversations if convers['from'] == 'user']
        score = 0.0
        for instruction in instructions:
            score += scorer.infer_complexity(instruction)
        results.append(score)
    return results


def infer_quality(dataset):
    model_name = "hkust-nlp/deita-quality-scorer"
    scorer = Llama_Scorer(model_name_or_path=model_name, is_vllm=False)

    results = []
    for sample in tqdm(dataset):
        conversations = sample['conversations']
        instructions = [convers['value'] for convers in conversations if convers['from'] == 'user']
        responses = [convers['value'] for convers in conversations if convers['from'] == 'assistant']
        score = 0.0
        for instruction, response in zip(instructions, responses):
            score += scorer.infer_quality(instruction, response)
        results.append(score)
    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
