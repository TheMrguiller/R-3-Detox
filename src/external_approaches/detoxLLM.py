from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import pandas as pd
import sys
import os
project_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
from tqdm import tqdm
import gc
import torch

import warnings
warnings.filterwarnings('ignore')
class DetoxLLM:
    def __init__(self, model_name: str = "UBC-NLP/DetoxLLM-7B"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt = "Rewrite the following toxic input into non-toxic version. Let's break the input down step by step to rewrite the non-toxic version. You should first think about the expanation of why the input text is toxic. Then generate the detoxic output. You must preserve the original meaning as much as possible.\nInput: {text}\n"

    def generate_paraphrase(self, texts: List[str]) -> List[str]:
        inputs = self.tokenizer([self.prompt.format(text=text) for text in texts], return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        lenghth_of_inputs = [len(input) for input in inputs['input_ids']]
        print(f"Lenght of input: {lenghth_of_inputs}")
        free, total = torch.cuda.mem_get_info(self.model.device)
        mem_used_MB = (total - free) / 1024 ** 2
        print(f"Available total memory: {total / 1024 ** 2}")
        print(f"Memory used by model:{mem_used_MB}")
        try:
            outputs = self.model.generate(**inputs, max_length=4096, num_return_sequences=1,do_sample=False)
            outputs = outputs.to("cpu")
            results = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            del inputs, outputs
            
            results = [self.process_response(result) for result in results]
            for idx,result in enumerate(results):
                if type(result) == str:
                    text = texts[idx]
                    print(f"Error in generating output: {text}")
                    input_ = self.tokenizer([self.prompt.format(text=texts[idx])], return_tensors="pt", padding=True, truncation=True).to(self.model.device)
                    output = self.model.generate(**input_, max_length=4096, num_return_sequences=1,do_sample=False,repetition_penalty=1.2)
                    output= output.to("cpu")
                    result = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    results[idx] = self.process_response(result)
                
            gc.collect()
            torch.cuda.empty_cache()
            return results
        except Exception as e:
            
            print(f"Error in generating output: {e}")
            for idx,text in enumerate(texts):
                print(f"Text: {text}, length: {lenghth_of_inputs[idx]}")
            return [None]*len(texts)
    def process_response(self, response):
        try:
            response_list = response.split("\n")
            # print(response_list)
            response_list = response_list[2:]
            # print(response_list)
            final = {
                "reasoning": response_list[0].split(": ")[1],
                "paraphrase": response_list[1].split(": ")[1]
            }
            return final
        except Exception as e:
            print(f"Error in processing response: {response}, error: {e}")
            return response

if __name__ == "__main__":
    df = pd.read_csv(project_path+"data/processed/final_few_shot_reasoning/few_shot_reasoning.csv")
    data_path = project_path+"data/processed/final_paraphrases/detoxllm/"
    # print(f"Dataset read from {project_path+'data/processed/dataset/dataset.csv'}")
    batch_size = 8
    print(f"Preparing the model")
    model = DetoxLLM()
    print(f"Model ready")
    df.drop(columns=["reasoning"],inplace=True)
    df = df[df["source"]!="non_toxic"]
    # df = df[df["source"]=="parallel_detoxification"]
    df.reset_index(drop=True,inplace=True)
    df["result"] = ""
    # df = df[:4]
    print(f"Generating paraphrases for {len(df)} toxic sentences")
    for idx in tqdm(range(0,len(df),batch_size),total=len(df)//batch_size):
        text = df["sentence"].iloc[idx:idx+batch_size].tolist()
        results = model.generate_paraphrase(text)
        for i,result in enumerate(results):
            # print(result)
            df.loc[idx+i, "result"] = str(result)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    df.to_csv(data_path+"detoxllm.csv",index=False)
        