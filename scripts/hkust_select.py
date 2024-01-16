import torch 
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

hidden_states_cache = []

# do not name this module with easy names like `select` to invoid any dependency error, name your script with more complicated names.
def main(
    scored_data_dir: str = "/workspace/lit-llama/data/meta-math/scored.pt",
    threshold: float = 0.95,
    target_dataset_size: int = 6000,
    target_dataset_output_path: str = "/workspace/lit-llama/data/meta-math/final-6k.pt"
):
    global hidden_states_cache

    # hyper parameters
    LLAMA2 = "/workspace/lit-llama/checkpoints/Llama-2-13b-hf"
    THRESHOLD = threshold

    model = AutoModelForCausalLM.from_pretrained(LLAMA2)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA2)

    dataset = torch.load(scored_data_dir)
    for sample in dataset:
        sample['evol'] = sample['complexity_score'] * sample['quality_score']
    # rank the samples by evol, descending
    dataset = sorted(dataset, key=lambda x: x['evol'], reverse=True)

    # half precision + cuda
    model = model.half().cuda()
    model.eval()
    print(model)
    # TODO: register a forward hook to get the hidden states
    model.lm_head.register_forward_hook(hook_forward_fn)

    target_dataset = []
    sentence_embedding_list = []
    
    pbar = tqdm(total=target_dataset_size, desc="Selecting samples")
    for sample in dataset:
        sentence_embedding = get_sentence_embedding(model, tokenizer, sample)
        # promise diversity in the selected dataset
        if check_similarity(sentence_embedding, sentence_embedding_list, THRESHOLD):
            target_dataset.append(sample)
            sentence_embedding_list.append(sentence_embedding)
            pbar.update(1)
        if len(target_dataset) >= target_dataset_size:
            break
    
    pbar.close()
    torch.save(target_dataset, target_dataset_output_path)


def get_sentence_embedding(model, tokenizer, sample):
    global hidden_states_cache

    local_cache = []
    for converse in sample['conversations']:
        prompt = converse['value']
        encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(model.device)
        with torch.no_grad():
            outputs = model(encoded_prompt)
        hidden_states = hidden_states_cache[-1]
        local_cache.append(hidden_states)
    cat_hidden_states = torch.cat(local_cache, dim=1)
    avg_hidden = torch.mean(cat_hidden_states, dim=1).view(-1)
    return avg_hidden.cpu().numpy()


def check_similarity(sentence_embedding, sentence_embedding_list, THRESHOLD=0.9):
    for sentence_embedding_ in sentence_embedding_list:
        if cos_dist(sentence_embedding, sentence_embedding_) > THRESHOLD:
            print(f"Too close similarity: {cos_dist(sentence_embedding, sentence_embedding_)}")
            return False
    return True


def hook_forward_fn(module, input, output):
    global hidden_states_cache
    hidden_states_cache.append(input[0].clone().detach().cpu())


def cos_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
