import os
import sys
project_path=os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
import pandas as pd
import argparse
from src.utils.vector_store_incorrect_paraphrase import Vector_store_Evaluation
import json
from tqdm import tqdm

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--max_examples",type=int,default=45,help="Number of examples to use")
    argparser.add_argument("--chunk_index",type=int,default=1,help="Index of the chunk to process")
    argparser.add_argument("--all_chunks",type=str,default="False",help="Process all chunks")
    max_examples = argparser.parse_args().max_examples
    chunk_index = argparser.parse_args().chunk_index
    all_chunks = argparser.parse_args().all_chunks

    vector_store = Vector_store_Evaluation(db_path=project_path+"data/vector_store/")
    df = pd.read_csv(project_path+"data/processed/final_few_shot_reasoning/few_shot_reasoning.csv")
    df.drop(columns=["reasoning"],inplace=True)
    df = df[df["source"]!="non_toxic"]
    df.reset_index(drop=True,inplace=True)
    #Division of the dataframe in 5 chunks
    number_of_chunks = 8
    if all_chunks == "False":
        chunk_size = len(df) // number_of_chunks
        remainder = len(df) % number_of_chunks
        chunks = []
        start = 0
        for i in range(number_of_chunks):
            end = start + chunk_size
            chunks.append(df.iloc[start:end])
            start = end
        if remainder != 0:
            end = start + remainder
            chunks.append(df.iloc[start:end])
        df = chunks[chunk_index]

    json_few_shot_examples = {}
    for index, row in tqdm(df.iterrows(),total=len(df),desc="Generating few shot examples"):
        paraphrase = row["paraphrase"]
        examples = vector_store.query(paraphrase=paraphrase,n_results=max_examples)
        json_few_shot_examples[index] = examples
    if not os.path.exists(project_path+"data/processed/precomputed_evaluation_examples/"):
        os.makedirs(project_path+"data/processed/precomputed_evaluation_examples/")
    if all_chunks == "True":
        with open(project_path+"data/processed/precomputed_evaluation_examples/precomputed_evaluation_examples_all_chunks.json","w") as f:
            json.dump(json_few_shot_examples,f)
    else:
        with open(project_path+f"data/processed/precomputed_evaluation_examples/precomputed_evaluation_examples_{str(chunk_index)}.json","w") as f:
            json.dump(json_few_shot_examples,f)
        
