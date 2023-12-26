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
    with open(params.lima_path) as f:
        primary_list = json.load(f)
    with open(params.alpaca_path) as f:
        secondary_list = json.load(f)
    
    # assert(len(primary_list) == len(secondary_list))  # should be the same length

    collected_responses = list()
    out_file = f"./out/benchmark/gpt4-judge-{datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')}.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    for p_dict, s_dict in tqdm(zip(primary_list, secondary_list)):
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
    aa("--alpaca_path", type=str, default="./out/lora/alpaca/alpaca-23-12-16-16-12-45.json")
    aa("--lima_path", type=str, default="./out/lora/lima/lima-23-12-16-16-12-35.json")
    return parser.parse_args()


prompt_temp = """As the judge in this scenario, your role is to critically evaluate two responses provided in relation to a specific question. Your evaluation should be based on the criteria of informativeness, relevance, and engagement. The response that performs better in these areas should be deemed as better.

Evaluation Criteria:
- Informativeness: The response should provide valuable, accurate, and comprehensive information.
- Relevance: The response should directly address the question or topic at hand without deviation.
- Engagement: The response should be presented in a manner that captures interest and encourages further thought or conversation.


Please respond starting with the specification of your evaluation.

More answer Guidelines:
- Response ends with **Primary** if the first response outperforms the second response according to the above criteria.
- Response ends with **Secondary** if the second response exceeds the first response according to the above criteria.
- Response ends with **Neither** if both responses perform equivalently, with no significant difference in quality based on the above criteria.


### Question:
{question}

### Primary:
{primary}

### Secondary:
{secondary}
"""


if __name__ == "__main__":
    main()
