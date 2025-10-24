import transformers
import torch
import os
project_path=os.path.abspath(__file__).split('src')[0]
import pandas as pd

class ToxicMetric:
    def __init__(self, model_name: str="unitary/toxic-bert", tokenizer: str="unitary/toxic-bert"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = transformers.pipeline("text-classification", model=model_name, tokenizer=tokenizer, device=device,return_all_scores=True,truncation=True)
        self.tokenizer = tokenizer
        self.device = device

    def pred(self, texts: list):
        predictions = self.model(texts)
        prediction=[]
        for i in range(len(predictions)):
            prediction.append(predictions[i][0]['score']) # First is toxic score
        return prediction

from tqdm import tqdm
if __name__ == "__main__":
    df = pd.read_csv(project_path+"data/processed/final_few_shot_reasoning/few_shot_reasoning.csv")
    df = df[df["source"]!="non_toxic"]
    df.dropna(subset=["paraphrase"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    metric = ToxicMetric()
    batch_size = 32
    for i in tqdm(range(0, len(df), batch_size), desc="Toxicity", total=len(df)//batch_size):
        # print(i)
        texts = df["paraphrase"][i:i+batch_size].tolist()
        values = metric.pred(texts)
        for idx in range(len(values)):
            df.loc[i+idx, "toxicity"] = values[idx]
        pass
        # df.loc[i:i+batch_size-1, "toxicity"] = values
    print("Toxicity calculated for paraphrases")
    APPDIA_df = df[df["source"]=="APPDIA"]
    print(f"APPDIA toxic mean: {APPDIA_df['toxicity'].mean()}")
    paradetox_df = df[df["source"]=="paradetox"]
    print(f"paradetox toxic mean: {paradetox_df['toxicity'].mean()}")
    parallel_detoxification_df = df[df["source"]=="parallel_detoxification"]
    print(f"parallel_detoxification toxic mean: {parallel_detoxification_df['toxicity'].mean()}")

    metric = ToxicMetric()
    batch_size = 32
    for i in tqdm(range(0, len(df), batch_size), desc="Toxicity", total=len(df)//batch_size):
        # print(i)
        texts = df["sentence"][i:i+batch_size].tolist()
        values = metric.pred(texts)
        for idx in range(len(values)):
            df.loc[i+idx, "toxicity"] = values[idx]
        pass
        # df.loc[i:i+batch_size-1, "toxicity"] = values
    print("Toxicity calculated for orginal sentences")
    APPDIA_df = df[df["source"]=="APPDIA"]
    print(f"APPDIA toxic mean: {APPDIA_df['toxicity'].mean()}")
    paradetox_df = df[df["source"]=="paradetox"]
    print(f"paradetox toxic mean: {paradetox_df['toxicity'].mean()}")
    parallel_detoxification_df = df[df["source"]=="parallel_detoxification"]
    print(f"parallel_detoxification toxic mean: {parallel_detoxification_df['toxicity'].mean()}")
