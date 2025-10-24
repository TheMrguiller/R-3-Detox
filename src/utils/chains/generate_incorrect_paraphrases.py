import sys
import os
project_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
from src.utils.llms.llm_vllm_offline_proxy import LLMVLLMOfflineProxy
from src.utils.llms.llm_huggingface_proxy import LLMHuggingFace
import pandas as pd
from typing import List
import yaml
from tqdm import tqdm
from langchain_core.output_parsers import JsonOutputParser
from src.utils.structured_output.incorrect_paraphase import IncorrectParaphraseGeneration
import re
class IncorrectParaphraseGenerator:
    def __init__(self,model_config:str):
        self.llm = LLMVLLMOfflineProxy(model_config)
        # self.llm = LLMHuggingFace(model_config)
        self.max_tokens = self.llm.config["max_tokens"]
        self.parser = JsonOutputParser(pydantic_object=IncorrectParaphraseGeneration)
        with open(project_path + "src/utils/llms/prompts/generate_incorrect_paraphrase.yaml", "r") as file:
            base_prompt = yaml.safe_load(file) 
        self.system_prompt = base_prompt["system_prompt"].replace("\\n","\n").replace("\\t","\t")
        self.user_prompt = base_prompt["user_prompt"].replace("\\n","\n").replace("\\t","\t")
    
    def extract_content_inside_braces(self,text):
    # Regular expression to capture the content inside the largest pair of curly braces
        match = re.search(r'\{(.*?)\}', text, re.DOTALL)
        if match:
            return '{' + match.group(1).strip() + '}'  # Return the matched content wrapped in {}
        return None
    def generate_incorrect_paraphrase(self, original_sentences:List[str],paraphrase_sentences:List[str],batch_size:int=32) -> List[str]:
        results = []
        for i in tqdm(range(0,len(original_sentences),batch_size)):
            batch_original_sentences = original_sentences[i:i+batch_size]
            batch_paraphrase_sentences = paraphrase_sentences[i:i+batch_size]
            system_prompts = [self.system_prompt+f"{self.parser.get_format_instructions()}"] * len(batch_original_sentences)
            user_prompts= []
            for j in range(len(batch_original_sentences)):
                user_prompts.append(self.user_prompt.format(original_sentence=batch_original_sentences[j].replace('"',"'"),non_toxic_sentence=batch_paraphrase_sentences[j].replace('"',"'")))
            result_batch = self.llm.query(system_prompts,user_prompts)
            for result in result_batch:
                parsed_result=self.extract_content_inside_braces(result)
                results.append(parsed_result)
        return results
    

if __name__ == "__main__":
    model_config = project_path + "src/utils/llms/configs/qwen2_5_7B.yaml"
    generator = IncorrectParaphraseGenerator(model_config)
    df= pd.read_csv(project_path + "data/processed/final_few_shot_reasoning/few_shot_reasoning.csv")
    output_path = project_path + "data/processed/incorrect_paraphrases/incorrect_paraphrases.csv"
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.drop(columns=["reasoning"],inplace=True)
    df = df[df["source"]!="non_toxic"]
    df = df.dropna(subset=["paraphrase"])
    df.reset_index(drop=True,inplace=True)
    # df = df[:8]
    original_sentences = df["sentence"].tolist()
    paraphrase_sentences = df["paraphrase"].tolist()
    results = generator.generate_incorrect_paraphrase(original_sentences,paraphrase_sentences,batch_size=8)
    df["incorrect_paraphrase"] = results
    df.to_csv(output_path,index=False)
    