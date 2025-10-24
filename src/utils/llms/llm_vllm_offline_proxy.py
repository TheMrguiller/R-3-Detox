import yaml
from vllm import LLM, SamplingParams,RequestOutput
import re
import torch
from typing import List, Dict
import os
import time
import warnings
import argparse
warnings.filterwarnings("ignore")
# import sys
# proyect_path = os.path.abspath(__file__).split('src')[0]
# sys.path.append(proyect_path)
# from vllm.sampling_params import GuidedDecodingParams
# from src.utils.structured_output.paraphrase_generation import ParaphraseGeneration

class LLMVLLMOfflineProxy:
    config = None
    llm = None

    def __init__(self, yaml_config_file,temperature=0.0,seed=42,top_p=1.0,num_return_sequences=1) -> None:
        self._read_yaml_config(yaml_config_file)
        # json_schema = ParaphraseGeneration.model_json_schema()
        # guided_decoding_params = GuidedDecodingParams(json=json_schema)
        # self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, n=num_return_sequences, max_tokens=self.config["max_tokens"],guided_decoding=guided_decoding_params)
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, n=num_return_sequences, max_tokens=self.config["max_tokens"])
        self.llm = LLM(model=self.config["model_name"], tokenizer=self.config["model_name"],seed=seed, 
                       trust_remote_code=True,dtype=torch.float16,max_seq_len_to_capture=self.config["max_tokens"],
                       max_model_len=self.config["max_tokens"],kv_cache_dtype="auto",max_num_seqs=1024)
        
           

    def _read_yaml_config(self, yaml_config_file):
        # read yaml config file into a dictionary

        with open(yaml_config_file, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)['config']
            except yaml.YAMLError as exc:
                print(exc)

    def obtain_clean_response(self, responses:List[RequestOutput])->str:
        # Some responses are in chinese, so we need to translate them
        responses_list = []
        for output in responses:
            response=output.outputs[0].text
            responses_list.append(response)
        return responses_list

    def change_parameters(self,temperature=0.0,do_sample=False,seed=42,top_p=1.0,num_return_sequences=1):
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p,seed=seed, n=num_return_sequences, max_tokens=self.config["max_tokens"])

    def postprocess_response(self, response:str)->dict:
        preprocess_response = {
            "reasoning": response,
            "output": response,
        }
        if "special_thought_tokens_start" in self.config and "special_thought_tokens_end" in self.config:
            token_start = self.config["special_thought_tokens_start"]
            token_end = self.config["special_thought_tokens_end"]
            # Escape special characters in the token to avoid regex issues
            thought_pattern = f"{re.escape(token_start)}(.*?){re.escape(token_end)}"
            thought_match = re.search(thought_pattern, response, re.DOTALL)
            if thought_match:
                preprocess_response["reasoning"] = thought_match.group(1).strip()
        if "special_output_tokens_start" in self.config or "special_output_tokens_end" in self.config:
            token_start = self.config["special_output_tokens_start"]
            token_end = self.config["special_output_tokens_end"]
            # Escape special characters in the token to avoid regex issues
            output_pattern = f"{re.escape(token_start)}(.*?){re.escape(token_end)}"
            output_match = re.search(output_pattern, response, re.DOTALL)
            if output_match:
                preprocess_response["output"] = output_match.group(1).strip()
        return preprocess_response
    
    def query(self, system_messages: List[str] = None, user_messages: List[str] = None) -> List[dict]:
        # Ensure the inputs are batched
        if user_messages is None or len(user_messages) == 0:
            raise ValueError("user_messages cannot be empty for batch generation.")
        
        # Handle default system message for the batch
        batch_histories = []
        if not system_messages:
            if "system_message" in self.config:
                system_messages = [self.config["system_message"]] * len(user_messages)
                for system_message, user_message in zip(system_messages, user_messages):
                    history = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ]
                    batch_histories.append(history)
            else:
                for user_message in user_messages:
                    history = [
                        {"role": "user", "content": user_message}
                    ]
                    batch_histories.append(history)
        else:
            for system_message, user_message in zip(system_messages, user_messages):
                history = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
                batch_histories.append(history)
        print(f"Sampling params: {self.sampling_params}")
        reponses=self.llm.chat(messages=batch_histories, sampling_params=self.sampling_params,use_tqdm=True)
        # Generate responses in batch
        responses = self.obtain_clean_response(reponses)
        # Postprocess each response in the batch
        return responses
    
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--config_name",default="marco-o1",choices=["marco-o1","openO1","qwq_preview","skywork-o1"],help="Name of the c to use")
    config_name= parser.parse_args().config_name
    print(f"Using config {config_name}")
    project_path=os.path.abspath(__file__).split('src')[0]
    config_path = project_path+f"src/utils/llms/configs/{config_name}.yaml"

    print("Loading LLM")
    llm = LLMVLLMOfflineProxy(config_path)
    user_message_1 = """Given a toxic sentence, your task is to perform a detailed reasoning process. First, analyze the words or phrases in the sentence that convey toxic behavior, explaining why they are toxic within the context. Then, determine the specific changes needed to rewrite the sentence in a non-toxic manner while preserving its original meaning. Provide a clear explanation of how these changes remove the toxicity. To support your reasoning, consider any external information provided, such as a non-toxic rewritten sentence, possible relevant words that could express toxicity (though not necessarily), and the label. However, do not acknowledge or reference the existence of this external information during your reasoning process. Do not generate code.
Toxic sentece: two lunatics trying to show whos got the bigger thing !
Relevant words: lunatics bigger thing
Label: Toxic
Paraphrase: Two unsuitable people trying to show who's better."""
    user_message_2 = """Given a toxic sentence, your task is to perform a detailed reasoning process. First, analyze the words or phrases in the sentence that convey toxic behavior, explaining why they are toxic within the context. Then, determine the specific changes needed to rewrite the sentence in a non-toxic manner while preserving its original meaning. Provide a clear explanation of how these changes remove the toxicity. To support your reasoning, consider any external information provided, such as a non-toxic rewritten sentence, possible relevant words that could express toxicity (though not necessarily), and the label. However, do not acknowledge or reference the existence of this external information during your reasoning process. Do not state that you have external information saying that is toxic, that you have some initial relevant words or that you have the final paraphrase sentence. Do not generate code.
Toxic sentece: clinton is a loser , twice .
Relevant words: loser
Label: Toxic
Paraphrase: Clinton lost twice."""
    user_messages = [user_message_1, user_message_2]

    start = time.time()
    print("Querying LLM")
    responses = llm.query(None, user_messages)
    end = time.time()
    print(f"Time taken batch: {end - start} seconds")
    for response in responses:
        print(response)
    
    print("Querying LLM")
    
    for user_message in user_messages:
        start = time.time()
        response = llm.query(None, [user_message])
        end = time.time()
        print(f"Time taken single: {end - start} seconds")
        for response in responses:
            print(response)