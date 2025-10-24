import os
import sys
project_path=os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
import pandas as pd
import argparse
from src.utils.chains.paraphrase_generation_with_error_fixing import ParaphraseExperimentOfflineWithErrorsFixing
from typing import List
import glob
from src.utils.chains.structured_output_enforcer import StructuredOutputEnforcer
from tqdm import tqdm
from transformers import AutoTokenizer
import yaml

def get_experiment_results(experiment:ParaphraseExperimentOfflineWithErrorsFixing,sentences:List[str],paraphrases:List[str],labels:List[int],shap_values:List[List[str]],sources:List[str],indexes:List[int],save_path:str,experiment_type:str):

    results = experiment.run_experiment(sentences=sentences,labels=labels,shap_values=shap_values,sources=sources,indexes=indexes,batch_size=64)
    # print(results)
    df_try = pd.DataFrame(columns=["source","sentence","label","shap_values","result"])
    df_try["sentence"] = sentences
    df_try["label"] = labels
    df_try["shap_values"] = shap_values
    df_try["source"] = sources
    df_try["paraphrase"] = paraphrases
    if None in results: #If the prompt was too long to handle it properly in the max tokens we just finish the experiment from here
        return False
    df_try["result"] = results
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df_try.index = indexes
    return df_try

def get_experiment_type_number_examples(file):
    source_file_name = file.split("/")[-1].split("experiment_")[1]
    source_file_name = source_file_name.split("_results")[0]
    if source_file_name!="zero_shot":
        experiment_type = source_file_name.rsplit("_",1)[0]
    else:
        experiment_type = "zero_shot"
    if experiment_type=="zero_shot":
        number_examples = 0
    else:
        number_examples = source_file_name.rsplit("_",1)[1]
        number_examples = int(number_examples)
    return experiment_type,number_examples

def row_dict_value_empty(row):
    if type(row)==dict:
        for key in row:
            if row[key]=="":
                return True
    return False
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--folder",type=str,help="Folder we want to extract the paraphrase from",default="qwen2_5_7B")#llama3_1_8B,openO1,marco-o1,qwen2_5_7B
    args.add_argument("--use_enforcer",type=str,help="Use a model that enforces the structure of the output",default="True")
    args.add_argument("--enforcer_config",type=str,help="Model configuration to use for the enforcer",default="qwen2_5")#phi_mini
    folder = args.parse_args().folder
    use_enforcer = args.parse_args().use_enforcer
    enforcer_config = args.parse_args().enforcer_config
    
    output_path = project_path+"data/processed/final_paraphrases/"+folder+"/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data_path = project_path+"data/interim/final_paraphrases_postprocess/"+folder+"/"
    files = glob.glob(data_path+"*.csv")
    number_of_non_correct_json = 0
    number_of_hallucinations = 0
    generation_model_config = project_path+f"src/utils/llms/configs/{folder}.yaml"
    enforcer_config = project_path+f"src/utils/llms/configs/{enforcer_config}.yaml"
    with open(generation_model_config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)['config']
        except yaml.YAMLError as exc:
            print(exc)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    files.sort(reverse=False)
    experiment_type,number_examples=get_experiment_type_number_examples(files[0])
    files = [files[1]]
    experiment = ParaphraseExperimentOfflineWithErrorsFixing(generation_model_config=generation_model_config,data_extractor_config=enforcer_config,experiment_type=experiment_type,num_examples=number_examples)
    # files.sort(reverse=True)
    for file in files:
        experiment_type,number_examples=get_experiment_type_number_examples(file)
        experiment.num_examples =number_examples
        experiment.experiment_type = experiment_type
        
        df = pd.read_csv(file)
        df["result"] = df["result"].apply(lambda x: eval(x) if "{" in x and "}" in x else x)
        # df["Correct"] = False
        number_of_hallucinations = 0
        df_with_hallucinations = df[(df["result"]=="Model hallucinated") | (df["result"]=="") | (df["result"].apply(row_dict_value_empty))]
        print(f"Experiment type: {experiment_type} with {number_examples} examples, total hallucinations: {len(df_with_hallucinations)}, model: {folder}")
        if len(df_with_hallucinations)==0:
            df.to_csv(output_path+f"experiment_{experiment_type}_{number_examples}_results.csv",index=False)
            continue
        print(f"Experiment type: {experiment_type} with {number_examples} examples, total hallucinations: {len(df_with_hallucinations)}, model: {folder}")
        sentences = df_with_hallucinations["sentence"].tolist()
        labels = df_with_hallucinations["label"].tolist()
        df_with_hallucinations["shap_values"]= df_with_hallucinations["shap_values"].apply(eval)
        shap_values = df_with_hallucinations["shap_values"].tolist()
        sources = df_with_hallucinations["source"].tolist()
        if "paraphrase" not in df_with_hallucinations.columns:
            df_with_hallucinations["paraphrase"] = ""
        paraphrases = df_with_hallucinations["paraphrase"].tolist()
        indexes = df_with_hallucinations.index.tolist()
        df_with_hallucinations=get_experiment_results(experiment=experiment,sentences=sentences,paraphrases=paraphrases,labels=labels,shap_values=shap_values,sources=sources,indexes=indexes,save_path=output_path,experiment_type=experiment_type)
        if type(df_with_hallucinations)!=bool:
            for index,row in df_with_hallucinations.iterrows():
                df.at[index,"result"] = row["result"]
            df.to_csv(output_path+f"experiment_{experiment_type}_{number_examples}_results.csv",index=False)

