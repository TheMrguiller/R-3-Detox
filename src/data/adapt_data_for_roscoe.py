import pandas as pd
import glob
import os
import yaml

def process_with_prompt(file):
    df = pd.read_csv(file)
    df_roscoe = pd.DataFrame(columns=["premise","gpt-3","hypothesis"])
    for idx, row in df.iterrows():
        if row["label"] == 1.0:
            prompt = toxic_prompt_template.format(sentence=row["sentence"],paraphrase=row["paraphrase"],toxic_words=row["shap_values"],label="Toxic")
        else:
            prompt = non_toxic_prompt_template.format(sentence=row["sentence"],paraphrase=row["paraphrase"],toxic_words=row["shap_values"],label="Non-toxic")
        df_roscoe.loc[len(df_roscoe)] = [prompt,row["reasoning"],""]
    return df_roscoe
if __name__ == '__main__':
    proyect_path = os.path.abspath(__file__).split('src')[0]
    with open(proyect_path + "src/utils/llms/prompts/generate_reasoning_prompt.yaml", "r") as file:
        prompt_config = yaml.safe_load(file)
    toxic_prompt_template = prompt_config["toxic_template"]
    non_toxic_prompt_template = prompt_config["non_toxic_template"]
    files = glob.glob(proyect_path + "data/processed/reasoning_human_eval/*.csv")
    if not os.path.exists(proyect_path + "src/evaluation/parlai-app/projects/roscoe/roscoe_data/human_eval/"):
        os.makedirs(proyect_path + "src/evaluation/parlai-app/projects/roscoe/roscoe_data/human_eval/")
    for file in files:
        model_name = file.split("reasoning_human_eval_")[1].split(".csv")[0]
        process_with_prompt(file).to_json(proyect_path + f"src/evaluation/parlai-app/projects/roscoe/roscoe_data/human_eval/reasoning_{model_name}.json", orient='records', lines=True)
        
            
