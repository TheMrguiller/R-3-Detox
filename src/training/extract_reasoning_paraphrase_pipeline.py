import sys
import os
project_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
from src.utils.chains.clean_reasoning import LLMReasoningCleaning
import glob
import pandas as pd
import argparse
from tqdm import tqdm

def clean_reasoning_process(text):

    if "**Final Solution**" in text:
        text = text.split("**Final Solution**")[0]
        return text
    if "## Final Solution" in text:
        text = text.split("## Final Solution")[0]
        return text
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract reasoning from LLMs')
    parser.add_argument('--data_dir', type=str, help='Path to the directory containing the reasoning',default=project_path + "data/interim/few_shot_reasoning/")
    parser.add_argument('--batch_size', type=int, help='Batch size for reasoning extraction',default=64)
    parser.add_argument('--number_reasonings', type=int, help='Number of reasonings to extract',default=5)
    args = parser.parse_args()
    data_dir = args.data_dir
    batch_size = args.batch_size
    number_reasonings = args.number_reasonings
    file = data_dir + "few_shot_reasoning_qwq_preview.csv"
    df = pd.read_csv(file)
    # df = df[:8]
    llm_reasoning_cleaning = LLMReasoningCleaning(project_path + "src/utils/llms/configs/qwen2_5_32B.yaml")
    df["reasoning_sampling"] = df["reasoning_sampling"].apply(eval)
    for index in tqdm(range(0,len(df),batch_size)):
        reasonings = []
        labels = []
        end = min(len(df), index+batch_size)
        for idx in range(index,end):
            row = df.iloc[idx]
            reasonings = reasonings + [clean_reasoning_process(row["reasoning_greedy"])] + [clean_reasoning_process(reasoning_sampling) for reasoning_sampling in row["reasoning_sampling"]]
            labels.extend([row["label"]]*(len(row["reasoning_sampling"])+1))
        clean_reasoning = llm_reasoning_cleaning.query(labels,reasonings,batch_size=len(reasonings))
        for idx in range(0,len(clean_reasoning),number_reasonings):
            df.loc[index+int(idx/number_reasonings),"reasoning_greedy"] = clean_reasoning[idx]
            df.at[index+int(idx/number_reasonings),"reasoning_sampling"] = clean_reasoning[idx+1:idx+number_reasonings]
    file = file.replace("interim","processed")
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    df.to_csv(file,index=False)
