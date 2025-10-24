import pandas as pd
import os
import sys
project_path=os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
import random
from tqdm import tqdm

class Format_data_structure:
    def __init__(self, data_path:str, model_name:str, best_model_name_path:str):
        self.df = pd.read_csv(data_path + model_name+"/"+best_model_name_path+".csv")
        self.colum_name =  "result"
        self.model_name = model_name
        self.df["result"] = self.df["result"].apply(lambda x: eval(x) if "{" in x and "}" in x else x )

def get_shuffled_list(original_list, seed):
    random.seed(seed)
    shuffled_list = original_list.copy()
    random.shuffle(shuffled_list)
    return shuffled_list

if __name__ == '__main__':
    # Load the data
    data_path = project_path + 'data/processed/final_paraphrases/'
    model_list = ["detoxllm","llama3_1_8B","marco-o1","openO1","qwen2_5_7B","qwq_preview","pseudoparadetox","paradetox","paradetox"]
    best_model_name = ["detoxllm",
    "experiment_few_shot_7_results","experiment_few_shot_10_results","experiment_few_shot_10_results",
    "experiment_few_shot_7_results","experiment_few_shot_10_results","pseudoparadetox","bart","bart"]
    # Bart will be use to select the reference paraphrases
    list_data = []
    for i in range(len(model_list)):
        list_data.append(Format_data_structure(data_path, model_list[i], best_model_name[i]))
    reference_dataset = list_data[-1].df
    list_data[-1].colum_name = "paraphrase"
    list_data[-1].model_name = "human"
    # prompt = "Provided a toxic sentence and words that may convey toxic meaning, your task is to generate a non-toxic rephrased version of the toxic sentence while maintaining the same meaning and matching the target style/attribute as much as possible.\nToxic sentence: {sentence}\nPossible toxic words: {toxic_words}"
    prompt = "Given a toxic sentence and words that may convey a toxic meaning, your task is to generate a non-toxic rephrased version of the sentence while maintaining the same meaning and matching the target style or attribute as closely as possible. A perfect non-toxic paraphrase is one that perfectly preserves the meaning, is inoffensive, and makes the least modifications to the original sentence. Avoid generating a paraphrase that either preserves the meaning but retains some offensiveness or is inoffensive but significantly alters the original meaning.\nToxic sentence: {sentence}\nPossible toxic words: {toxic_words}"
    df = pd.DataFrame(columns=["idx","prompt","modelAparaphrase","modelBparaphrase","modelCparaphrase","modelA","modelB","modelC","source"])
    for index in tqdm(range(len(reference_dataset))):
        new_list = get_shuffled_list(list_data, index)
        triplets = [new_list[i:i+3] for i in range(0, len(new_list), 3)]
        for triplet in triplets:
            prompt_ = prompt.format(sentence=reference_dataset.loc[index]["sentence"], toxic_words=reference_dataset.loc[index]["shap_values"])
            value0 = triplet[0].df.loc[index][triplet[0].colum_name]
            value1 = triplet[1].df.loc[index][triplet[1].colum_name]
            value2 = triplet[2].df.loc[index][triplet[2].colum_name]
            value0 = value0["paraphrase"] if type(value0) == dict else value0
            value1 = value1["paraphrase"] if type(value1) == dict else value1
            value2 = value2["paraphrase"] if type(value2) == dict else value2
            model_name0 = triplet[0].model_name
            model_name1 = triplet[1].model_name
            model_name2 = triplet[2].model_name
            if type(value0)==float or type(value1)==float or type(value2)==float:
                # print(f"Value 0: {value0}, Value 1: {value1}, Value 2: {value2}")
                continue
            df.loc[len(df)] = [
                index,
                prompt_,
                value0,
                value1,
                value2,
                model_name0,
                model_name1,
                model_name2,
                triplet[0].df.loc[index]["source"]
            ]
    df_copy = pd.DataFrame(columns=["idx","prompt","modelAparaphrase","modelBparaphrase","modelCparaphrase","modelA","modelB","modelC","source"])
    df_grouped = df.groupby('idx')
    for group in tqdm(df_grouped):
        if len(group[1])!=3:
            continue
        for i in range(3):
            df_copy.loc[len(df_copy)] = [group[1].iloc[i]['idx'],group[1].iloc[i]['prompt'],group[1].iloc[i]['modelAparaphrase'],group[1].iloc[i]['modelBparaphrase'],group[1].iloc[i]['modelCparaphrase'],group[1].iloc[i]['modelA'],group[1].iloc[i]['modelB'],group[1].iloc[i]['modelC'],group[1].iloc[i]['source']]
    
    df = df_copy
    df.reset_index(drop=True, inplace=True)

    output_path = project_path + "data/interim/paraphrasing_eval_dataset/paraphrasing_evaluation_dataset.csv"
    output_path_human_eval = project_path + "data/interim/human_eval_paraphrasing/paraphrasing_human_evaluation_dataset.csv"
    if not os.path.exists(project_path + "data/interim/paraphrasing_eval_dataset/"):
        os.makedirs(project_path + "data/interim/paraphrasing_eval_dataset/")
    if not os.path.exists(project_path + "data/interim/human_eval_paraphrasing/"):
        os.makedirs(project_path + "data/interim/human_eval_paraphrasing/")
    df.to_csv(output_path, index=False)
    
    df_APPDIA = df[df["source"] == "APPDIA"]
    df_paradetox = df[df["source"] == "paradetox"]
    df_parallel_detoxification = df[df["source"] == "parallel_detoxification"]
    df_APPDIA.reset_index(drop=True, inplace=True)
    df_paradetox.reset_index(drop=True, inplace=True)
    df_parallel_detoxification.reset_index(drop=True, inplace=True)
    idx_APPDIA = set(df_APPDIA["idx"].to_list())
    idx_paradetox = set(df_paradetox["idx"].to_list())
    idx_parallel_detoxification = set(df_parallel_detoxification["idx"].to_list())
    random_idx_APPDIA = random.sample(idx_APPDIA,17)
    random_idx_paradetox = random.sample(idx_paradetox,17)
    random_idx_parallel_detoxification = random.sample(idx_parallel_detoxification,17)
    df_APPDIA = df_APPDIA[df_APPDIA["idx"].isin(random_idx_APPDIA)]
    df_paradetox = df_paradetox[df_paradetox["idx"].isin(random_idx_paradetox)]
    df_parallel_detoxification = df_parallel_detoxification[df_parallel_detoxification["idx"].isin(random_idx_parallel_detoxification)]
    df_human_eval = pd.concat([df_APPDIA, df_paradetox, df_parallel_detoxification])
    df_human_eval.reset_index(drop=True, inplace=True)
    df_human_eval.to_csv(output_path_human_eval, index=False)