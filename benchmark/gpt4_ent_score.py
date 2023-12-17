import json
import argparse
import time
import datetime
import os

import openai
from tqdm import tqdm
from datasets import load_dataset

from utils import get_gpt_response


def main():
    params = parse_args()
    train_dataset = load_dataset(params.data_path, split="train")

    collected_responses = list()
    for sample in tqdm(train_dataset):
        prompt = prompt_temp.format(example="\n".join(sample['conversations']))  # join the conversations with newlines
        messages = [{"role": "user", "content": prompt}]
        response = get_gpt_response(params, messages=messages)
        collected_responses.append(dict(
            prompt=prompt,
            response=response,
        ))
        time.sleep(1)
        break 
    
    out_file = f"./out/benchmark/gpt4-{params.data_path.replace('/', '-')}-{datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')}.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(collected_responses, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    # --- model params --- #
    aa("--model_name", type=str, default="gpt-4-1106-preview")  # gpt-4-0314
    aa("-t", "--temperature", type=float, default=0)
    aa("--max_tokens", type=int, default=4096)
    aa("--top_p", type=float, default=0.9)
    aa("--data_path", type=str, default="GAIR/lima")
    return parser.parse_args()


prompt_temp = """Your task is to evaluate and rate the need for diversity in dialogue responses on a scale from 0 to 99. Diversity in this context refers to the variety of topics, expressions, and the complexity of language used in the dialogue. A higher score indicates a greater need for diverse, flexible, and nuanced responses, while a lower score signifies a context where precise, deterministic, and unambiguous responses are required.

Please use the following guidelines when rating:

- Score higher (70-99) for dialogue that occurs in:
  - Varied and dynamic social contexts.
  - Everyday conversations with a wide range of possible topics.
  - Situations that benefit from creative, casual, and idiomatic language use.

- Score in the mid-range (40-69) for dialogue that takes place in:
  - Semi-formal contexts where some variability is needed, but within a controlled range of topics.
  - Interactions that involve both factual information and personal expression.

- Score lower (0-39) for dialogue that is found in:
  - Highly technical, scientific, or specialized settings.
  - Scenarios requiring strict accuracy and precision, such as mathematics or coding.
  - Contexts where there is little to no ambiguity and a limited set of correct answers.

#### Example:
{example}
"""

if __name__ == "__main__":
    main()
