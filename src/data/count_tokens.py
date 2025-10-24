import glob

import pandas
import os
from transformers import AutoTokenizer
project_path=os.path.abspath(__file__).split('src')[0]

def count_tokens(text, tokenizer):
    return len(tokenizer.tokenize(text))
if __name__ == '__main__':
    data = []
    tokenizer = AutoTokenizer.from_pretrained(
                "BAAI/JudgeLM-33B-v1.0",
                use_fast=False,
                revision="main"
            )
    print(f"Hi")
    for file in glob.glob(project_path + '/data/processed/reasoning_human_eval/*.csv'):
        df = pandas.read_csv(file)
        df['tokens'] = df['reasoning'].apply(lambda x: count_tokens(x, tokenizer))
        file = os.path.basename(file)
        print(f"Number of texts with more than 1000 tokens in {file}: {len(df[df['tokens'] > 1000])}")
        print(f"Number of texts with more than 900 tokens in {file}: {len(df[df['tokens'] > 900])}")
        print(f"Number of texts with more than 800 tokens in {file}: {len(df[df['tokens'] > 800])}")
        print(f"Number of texts with more than 700 tokens in {file}: {len(df[df['tokens'] > 700])}")
        print(f"Number of texts with more than 600 tokens in {file}: {len(df[df['tokens'] > 600])}")
        
