import json
import argparse
import time
import datetime
import os

import openai
from tqdm import tqdm

from utils import get_gpt_response


def main():
    params = parse_args()
    with open(params.first_path) as f:
        primary_list = json.load(f)
    with open(params.second_path) as f:
        secondary_list = json.load(f)
    
    # assert(len(primary_list) == len(secondary_list))  # should be the same length

    collected_responses = list()
    out_file = f"./out/benchmark/gpt4-judge-{datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')}.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    pairs = list(zip(primary_list, secondary_list))
    if params.max_turns is not None:
        paris = pairs[:params.max_turns]
    for p_dict, s_dict in tqdm():
        assert(p_dict["inputs"] == s_dict["inputs"])
        
        question = p_dict["inputs"]
        primary = p_dict["response"]
        secondary = s_dict["response"]
        prompt = prompt_temp.format_map(dict(
            question=question,
            primary=primary,
            secondary=secondary,
        ))  # join the conversations with newlines
        messages = [{"role": "user", "content": prompt}]
        response = get_gpt_response(params, messages=messages)
        collected_responses.append(dict(
            prompt=prompt,
            response=response,
        ))
        
        time.sleep(1)
        
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
    aa("--first_path", type=str, default="./out/lora/alpaca/alpaca-23-12-16-16-12-45.json")
    aa("--second_path", type=str, default="./out/lora/lima/lima-23-12-16-16-12-35.json")
    aa("--max_turns", type=int, default=None)
    return parser.parse_args()


prompt_temp = """
As the judge in this scenario, your role is to critically evaluate two responses provided in relation to a specific question. Your evaluation should be based on the criteria of instruction following, hallucination, and informativeness. The response that performs better in these areas should be deemed as better.

**Evaluation Criteria:**

- **Instruction Following:** The response should directly and effectively address the question or task given without veering off-topic or providing unrelated information. It should align with the intention behind the question.

- **Hallucination:** The response must be free from fabrications and inaccuracies. It should provide information that is correct and verifiable, using the details provided in the question appropriately without introducing false or misleading content.

- **Informativeness:** Beyond being accurate, the response should offer enriched and valuable information that contributes to a deep understanding of the topic or provides practical and actionable solutions.

Based on these criteria, assess the following two answers to determine if:

1. The first answer is significantly better than the second one.
2. Neither answer is significantly better or worse than the other.
3. The first answer is significantly worse than the second one.

Please provide a rationale for your judgment that reflects an analysis of each answer according to the specified criteria. And end your response with a clear choice of your judgment from **1**, **2** and **3**.


### Question:
{question}

### First answer:
{primary}

### Second answer:
{secondary}
"""


if __name__ == "__main__":
    main()
