import torch 

from pathlib import Path
from model_utils import Llama_Scorer
from tqdm import tqdm


def main(
    data_dir,
    output_dir = None,
):
    dataset = torch.load(data_dir)

    if output_dir is None:
        output_dir = Path(data_dir).parent / "scored_dataset.pt"
    
    complexity_score = infer_complexity(dataset)
    quality_score = infer_quality(dataset)

    # sanity check
    assert len(complexity_score) == len(quality_score) == len(dataset)

    for i, sample in enumerate(dataset):
        sample['complexity_score'] = complexity_score[i]
        sample['quality_score'] = quality_score[i]
    
    torch.save(dataset, output_dir)


def infer_complexity(dataset):
    model_name = "hkust-nlp/deita-complexity-scorer"
    scorer = Llama_Scorer(model_name_or_path=model_name, is_vllm=False)

    results = []
    for sample in tqdm(dataset):
        conversations = sample['conversations']
        instructions = [convers['instruction'] for convers in conversations if convers['from'] == 'human']
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
        instructions = [convers['instruction'] for convers in conversations if convers['from'] == 'human']
        responses = [convers['response'] for convers in conversations if convers['from'] == 'assistant']
        score = 0.0
        for instruction, response in zip(instructions, responses):
            score += scorer.infer_quality(instruction, response)
        results.append(score)
    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
