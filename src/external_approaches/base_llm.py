import sys
import os
project_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
from src.utils.llms.llm_huggingface_proxy import LLMHuggingFace
from src.utils.llms.llm_vllm_offline_proxy import LLMVLLMOfflineProxy
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import gc
import torch

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate paraphrases using a base LLM")
    parser.add_argument("--model_name", default="llama3_1_8B",type=str, required=True, help="Name of the model to use")
    
    args = parser.parse_args()
    model_name = args.model_name
    model_config =project_path+f"src/utils/llms/configs/{model_name}.yaml"
    df = pd.read_csv(project_path+"data/processed/final_few_shot_reasoning/few_shot_reasoning.csv")
    data_path = project_path+"data/processed/final_paraphrases/basellm/"+model_name+"/"
    # print(f"Dataset read from {project_path+'data/processed/dataset/dataset.csv'}")
    batch_size = 8
    print(f"Preparing the model")
    if torch.cuda.is_available():
        llm = LLMVLLMOfflineProxy(model_config)
    else:
        llm = LLMHuggingFace(model_config)
    print(f"Model ready")
    df.drop(columns=["reasoning"],inplace=True)
    df = df[df["source"]!="non_toxic"]
    # df = df[df["source"]=="parallel_detoxification"]
    df.reset_index(drop=True,inplace=True)
    df["result"] = ""
    prompt = "Your task is text style/attribute transfer. Rewrite the above text into toxic/non-toxic language. You must match the target style/attribute and preserve the original meaning as much as possible. Do not explain the response. Do not hallucinate or add anything beyond the original input text. Do not include the input text in the response. Only generate the target text.\nInput Text: {text}"
    # df = df[:4]
    print(f"Generating paraphrases for {len(df)} toxic sentences")
    for idx in tqdm(range(0,len(df),batch_size),total=len(df)//batch_size):
        text = df["sentence"].iloc[idx:idx+batch_size].tolist()
        text = [prompt.format(text=text) for text in text]
        results = llm.query(user_messages=text)
        for i,result in enumerate(results):
            # print(result)
            df.loc[idx+i, "result"] = str(result)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    print(f"Saving results to {data_path+model_name}.csv")
    df.to_csv(data_path+model_name+".csv",index=False)
