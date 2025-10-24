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
    parser.add_argument('--batch_size', type=int, help='Batch size for reasoning extraction',default=256)
    args = parser.parse_args()
    data_dir = args.data_dir
    batch_size = args.batch_size
    # files = ["few_shot_reasoning_openO1.csv","few_shot_reasoning_marco-o1.csv","few_shot_reasoning_qwq_preview.csv"]
    files = ["few_shot_reasoning_openO1_left_no_toxic.csv","few_shot_reasoning_marco-o1_left_no_toxic.csv","few_shot_reasoning_qwq_preview_left_no_toxic.csv"]
    files = [data_dir + file for file in files]
    llm_reasoning_cleaning = LLMReasoningCleaning(project_path + "src/utils/llms/configs/qwen2_5_32B.yaml")
    # file = data_dir + "few_shot_reasoning_qwq_preview.csv"
    for file in files:
        df = pd.read_csv(file)
        # df = df[:8]
        for index in tqdm(range(0,len(df),batch_size)):
            end = min(len(df), index + batch_size)
            reasonings = df.loc[index:end, "reasoning"].tolist()
            labels = df.loc[index:end, "label"].tolist()
            clean_reasoning = llm_reasoning_cleaning.query(labels,reasonings,batch_size=batch_size)
            for idx in range(len(clean_reasoning)):
                df.loc[index+idx,"reasoning"] = clean_reasoning[idx]
        file = file.replace("interim","processed")
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        df.to_csv(file,index=False)
