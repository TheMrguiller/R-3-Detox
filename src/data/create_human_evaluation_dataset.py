import pandas as pd
import glob
import os
import yaml

if __name__ == "__main__":
    proyect_path = os.path.abspath(__file__).split('src')[0]
    with open(proyect_path + "src/utils/llms/prompts/generate_reasoning_prompt.yaml", "r") as file:
        prompt_config = yaml.safe_load(file)
    toxic_prompt_template = prompt_config["toxic_template"]
    non_toxic_prompt_template = prompt_config["non_toxic_template"]
    files = glob.glob(proyect_path + "data/processed/reasoning_human_eval/*.csv")
    dfs = [pd.read_csv(file) for file in files]
    model_name = [file.split("reasoning_human_eval_")[1].split(".csv")[0] for file in files]
    df_final = pd.DataFrame(columns=["idx","modelAreasoning","modelBreasoning","modelA","modelB"])
    for index in range(len(dfs[0])):
        label = dfs[0].loc[index]["label"]
        if label == 1.0:
            prompt =toxic_prompt_template.format(sentence=dfs[0].loc[index]["sentence"],paraphrase=dfs[0].loc[index]["paraphrase"],toxic_words=dfs[0].loc[index]["shap_values"],label="Toxic")
        else:
            prompt =non_toxic_prompt_template.format(sentence=dfs[0].loc[index]["sentence"],paraphrase=dfs[0].loc[index]["paraphrase"],toxic_words=dfs[0].loc[index]["shap_values"],label="Non-toxic")
        df_comment = pd.DataFrame(columns=["idx", "modelAreasoning", "modelBreasoning", "modelA", "modelB","prompt"])
        for start in range(0,len(dfs) - 1):
            for i in range(start, len(dfs) - 1):
                df_comment.loc[len(df_comment)] = [
                    f"D{index}",
                    dfs[start].loc[index]["reasoning"],
                    dfs[i + 1].loc[index]["reasoning"],
                    model_name[start],
                    model_name[i + 1],
                    prompt
                ]
        # Shuffle and reset index
        df_comment = df_comment.sample(frac=1, random_state=index).reset_index(drop=True)
        df_final = pd.concat([df_final, df_comment])
    if not os.path.exists(proyect_path + "data/interim/human_eval_reasoning/"):
        os.makedirs(proyect_path + "data/interim/human_eval_reasoning/")
    df_final.to_csv(proyect_path + "data/interim/human_eval_reasoning/human_eval_dataset.csv", index=False)
            
    

     