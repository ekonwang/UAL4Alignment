import json
import argparse
import time
import datetime
import os
import sys

from tqdm import tqdm
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from benchmark.utils import get_gpt_response


def main():
    params = parse_args()
    with open(params.first_path) as f:
        primary_list = json.load(f)
    with open(params.second_path) as f:
        secondary_list = json.load(f)
    
    # assert(len(primary_list) == len(secondary_list))  # should be the same length

    collected_responses = list()
    out_file = f"./out/benchmark/gpt4_judge/gpt4_judge_{datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')}_{params.tag}.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    pairs = list(zip(primary_list, secondary_list))
    if params.max_turns is not None:
        pairs = pairs[:params.max_turns]
    for p_dict, s_dict in tqdm(pairs):
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
        print(response)
        
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
    aa("--tag", type=str, required=True)
    return parser.parse_args()


prompt_temp = """
As the judge in this scenario, your role is to critically evaluate two responses provided in relation to a specific question. Your evaluation should be based on the following criteria. The response that performs better in these areas should be deemed as better.

**Evaluation Guidelines:**

1. **Adherence to Instructions:** The response must address the posed question or task directly and efficaciously, ensuring that the content remains on topic and eschews the inclusion of extraneous or irrelevant material. It must accurately reflect the original intent of the inquiry.

2. **Helpfulness**: The response should be informative and useful, providing a helpful to the question. It should be relevant to the question and provide a response that is both accurate and complete.

3. **Absence of Hallucination:** The response should be devoid of any fictitious or erroneous statements. It is imperative that the information provided is factual, corroborable, and makes appropriate use of the details presented in the question, without introducing any deceptive or spurious elements.

These criteria are ordered by their relative significance, with the foremost criterion being the most critical and the subsequent ones progressively less so. Using these guidelines, please evaluate the ensuing two answers to ascertain which is preferable:

1. The first answer is better than the second one.
2. Neither the first answer is better nor worse than the other.
3. The first answer is worse than the second one.

Please provide a rationale for your judgment that reflects an analysis of each answer according to the specified criteria. And end your response with a clear choice of your judgment from **1**, **2** and **3**. In the following format:

```
### Rationale:
[...]

### Judgment:
[...]
```


### Question:
{question}

### First answer:
{primary}

### Second answer:
{secondary}
"""


if __name__ == "__main__":
    main()
