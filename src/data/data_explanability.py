import sys
import os
project_path=os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
from src.models.toxicity_explainer import ToxicityExplainer
import pandas as pd
import re
import argparse
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def process_interesting_tokens(row):
    sentence = row["sentence"]
    sentence= re.findall(r'\w+|[^\w\s]', sentence)
    
    pass
def process_explanations(df:pd.DataFrame,explanations)->pd.DataFrame:
    logger.info("Processing explanations")
    df["shap_values"] = [explanation["shap_values"] for explanation in explanations]
    df["prediction"] = [explanation["prediction"] for explanation in explanations]
    df["tokenized_text"] = [explanation["tokenized_text"] for explanation in explanations]
    return df


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--model_name",default="tomh/toxigen_hatebert")
    model_name= parser.parse_args().model_name
    print(f"Explaining model {model_name}")
    df = pd.read_csv(project_path+"data/processed/dataset/dataset.csv")
    explainer = ToxicityExplainer(model_name,batch_size=1024)
    # df = df[df["source"]=="parallel_detoxification"]
    df_base = pd.read_csv(project_path+"data/processed/shap_values_aggregated/processed_shap_values.csv")
    df = df[~df["sentence"].isin(df_base["sentence"])]
    df = df[df["source"]=="non_toxic"]
    logger.info("Len of dataset:"+str(len(df)))
    explanations = explainer.explain(df["sentence"].to_list())
    logger.info("Explaining dataset done")
    df = process_explanations(df,explanations)
    model_name = model_name.replace("/","_")
    if not os.path.exists(project_path+f"data/interim/shap_values/"):
        os.makedirs(project_path+f"data/interim/shap_values/")
    df.to_csv(project_path+f"data/interim/shap_values/dataset_{model_name}.csv",index=False)
    logger.info("Saving dataset done")  