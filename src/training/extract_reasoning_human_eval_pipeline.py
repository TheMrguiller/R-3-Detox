import sys
import os
project_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
from src.utils.chains.clean_reasoning import LLMReasoningCleaning
import glob
import pandas as pd
import argparse

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
    parser.add_argument('--data_dir', type=str, help='Path to the directory containing the reasoning',default=project_path + "data/interim/reasoning_human_eval/")
    parser.add_argument('--batch_size', type=int, help='Batch size for reasoning extraction',default=4)
    args = parser.parse_args()
    data_dir = args.data_dir
    batch_size = args.batch_size
    files = glob.glob(data_dir+"*.csv")
    llm_reasoning_cleaning = LLMReasoningCleaning(project_path + "src/utils/llms/configs/qwen2_5.yaml")
    for file in files:
        print(file)
        df = pd.read_csv(file)
        df["reasoning"] = df["reasoning"].apply(clean_reasoning_process)
        df['reasoning'] = llm_reasoning_cleaning.query(df['label'].tolist(),df['reasoning'].tolist(),batch_size=batch_size)
        file = file.replace("interim","processed")
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        df.to_csv(file,index=False)
