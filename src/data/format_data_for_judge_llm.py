import pandas as pd
import os
import glob
import yaml
import json
import argparse
from typing import List
project_path=os.path.abspath(__file__).split('src')[0]

def formatting_function_reasoning(data:pd.DataFrame,toxic_template:str,non_toxic_template:str,model_name:str,save_folder:str):
   with open(f'{save_folder}/judgellm_{model_name}.json', 'w') as f:
    for index, row in data.iterrows():
        if row["label"] == 1.0:
            prompt =toxic_template.format(sentence=row["sentence"],paraphrase=row["paraphrase"],toxic_words=row["shap_values"],label="Toxic")
        else:
            prompt =non_toxic_template.format(sentence=row["sentence"],paraphrase=row["paraphrase"],toxic_words=row["shap_values"],label="Non-toxic")
        
        f.write(json.dumps({"question_id":index,"question_body":prompt,"model":model_name,"text":row["reasoning"]})+"\n")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--what_data", type=str, default="reasoning_eval",choices=["human_eval_reasoning","reasoning_eval","paraphrase_generation"])
    if argparser.parse_args().what_data=="human_eval_reasoning":
        with open(project_path + "src/utils/llms/prompts/generate_reasoning_prompt.yaml", "r") as file:
            prompt_config = yaml.safe_load(file)
        toxic_prompt_template = prompt_config["toxic_template"].replace("\\n","\n").replace("\\t","\t")
        non_toxic_prompt_template = prompt_config["non_toxic_template"].replace("\\n","\n").replace("\\t","\t")
        files = glob.glob(project_path + "data/processed/reasoning_human_eval/*.csv")
        save_folder= project_path + "data/interim/judgellm_reasoning_human_eval"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for file in files:
            data=pd.read_csv(file)
            model_name = file.split("reasoning_human_eval_")[1].split(".csv")[0]
            formatting_function_reasoning(data,toxic_prompt_template,non_toxic_prompt_template,model_name,save_folder)
    elif argparser.parse_args().what_data=="reasoning_eval":
        with open(project_path + "src/utils/llms/prompts/generate_reasoning_prompt.yaml", "r") as file:
            prompt_config = yaml.safe_load(file)
        toxic_prompt_template = prompt_config["toxic_template"].replace("\\n","\n").replace("\\t","\t")
        non_toxic_prompt_template = prompt_config["non_toxic_template"].replace("\\n","\n").replace("\\t","\t")
        files = glob.glob(project_path + "data/processed/few_shot_reasoning/*.csv")
        # files.pop(files.index(project_path + "data/processed/few_shot_reasoning/few_shot_reasoning_qwq_preview_sampling.csv"))
        
        save_folder= project_path + "data/interim/judgellm_few_shots_reasoning_eval/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for file in files:
            data=pd.read_csv(file)
            model_name = file.split("few_shot_reasoning_")[1].split(".csv")[0]
            formatting_function_reasoning(data,toxic_prompt_template,non_toxic_prompt_template,model_name,save_folder)
        
        
