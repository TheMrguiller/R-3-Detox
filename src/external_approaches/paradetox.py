from transformers import BartForConditionalGeneration, AutoTokenizer
from typing import List
import pandas as pd
import sys
import os
project_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
from tqdm import tqdm
class Paradetox:
    def __init__(self, model_name: str="s-nlp/bart-base-detox",tokenizer_name: str="facebook/bart-base"):
        self.model = BartForConditionalGeneration.from_pretrained(model_name,device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def generate_paraphrase(self, text: List[str]) -> List[str]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        outputs = self.model.generate(**inputs, max_length=1024, num_return_sequences=1,do_sample=False)
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
if __name__=="__main__":
    df = pd.read_csv(project_path+"data/processed/final_few_shot_reasoning/few_shot_reasoning.csv")
    data_path = project_path+"data/processed/final_paraphrases/paradetox/"
    batch_size = 32
    model = Paradetox()
    df.drop(columns=["reasoning"],inplace=True)
    df = df[df["source"]!="non_toxic"] 
    df.reset_index(drop=True,inplace=True)
    df["result"] = ""
    for idx in tqdm(range(0,len(df),batch_size),total=len(df)//batch_size):
        text = df["sentence"].iloc[idx:idx+batch_size].tolist()
        results = model.generate_paraphrase(text)
        for i,result in enumerate(results):
            df.loc[idx+i, "result"] = result
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    df.to_csv(data_path+"bart.csv",index=False)
    
    
        