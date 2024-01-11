import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class Scorer(object):
    
    def __init__(self, model_name_or_path: str, is_vllm: bool  = False):
        
        self.is_vllm = is_vllm
        
        if not is_vllm:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        else:
            
            from vllm import LLM, SamplingParams
            
            self.llm = LLM(model_name_or_path)
            self.sampling_params = SamplingParams(max_tokens = 512, logprobs = 1000)
        
    def infer_score(self, user_input: str):

        max_length = 512
        
        if self.is_vllm:
            outputs = self.llm.generate(user_input, self.sampling_params)
            score_template = np.array([1,2,3,4,5,6])
            
            try:
                logprobs_list = outputs[0].outputs[0].logprobs[0]
            except IndexError:
                logger.warning("Meeting Index Error. Returning A Placeholder Score -1.")
                return -1
        else:
            input_ids = self.tokenizer.encode(user_input, return_tensors = "pt")
            outputs = self.model.generate(input_ids, max_length = max_length, num_return_sequences = 1, return_dict_in_generate = True, output_scores = True)
            logprobs_list = outputs.scores[0][0]
            
        score_logits = []
        score_template = np.array([1,2,3,4,5,6])
        for k in self.id2score:
            score_logits.append(logprobs_list[k])
        score_logits = np.array(score_logits)
        score_npy = softmax(score_logits, axis=0)
        score_npy = score_npy * score_template

        score_npy = np.sum(score_npy, axis=0)
        
        return score_npy
            
    def infer_complexity(self, input_text: str):
        
        complexity_template = self.complexity_template
        user_input = complexity_template.format(instruction=input_text)
        
        return self.infer_score(user_input)
        
    def infer_quality(self, input_text: str, resp_text: str):
        
        quality_template = self.quality_template
        user_input = quality_template.format(instruction=input_text, output=resp_text)
        
        return self.infer_score(user_input)

    @property
    def id2score(self):
        raise NotImplementedError
    
    @property
    def complexity_template(self):
        raise NotImplementedError
    
    @property
    def quality_template(self):
        raise NotImplementedError


class Llama_Scorer(Scorer):
    
    @property
    def id2score(self):
        
        id2score = {
                29896: "1",
                29906: "2",
                29941: "3",
                29946: "4",
                29945: "5",
                29953: "6"
                }
        
        return id2score
    
    @property
    def complexity_template(self):
        
        complexity_template = ("You are a helpful assistant. Please identify the complexity score of the following user query. \n##Query: {instruction}  \n##Complexity: ")
        
        return complexity_template
    
    @property
    def quality_template(self):
        
        quality_template = ("You are a helpful assistant. Please identify the quality score of the Response corresponding to the Question. \n #Question#:\n{instruction}\n#Response#:\n{output} \n##Quality: ")
        
        return quality_template
