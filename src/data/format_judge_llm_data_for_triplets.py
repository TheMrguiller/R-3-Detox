import pandas as pd
import os
import sys
project_path=os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
import argparse
from tqdm import tqdm

def generate_triplets_evaluation(df:pd.DataFrame,output_path:str):
    df_grouped = df.groupby('idx')
    len_grouped = len(df_grouped.groups[0])

    for i in tqdm(range(len_grouped)):
        df_json = pd.DataFrame(columns=['question_id','question_body','answer1_body','answer2_body','answer3_body','answer1_model_id','answer2_model_id','answer3_model_id'])
        for group in tqdm(df_grouped):
            df_json.loc[len(df_json)] = [group[1].iloc[i]['idx'],group[1].iloc[i]['prompt'],group[1].iloc[i]['modelAparaphrase'],group[1].iloc[i]['modelBparaphrase'],group[1].iloc[i]['modelCparaphrase'],group[1].iloc[i]['modelA'],group[1].iloc[i]['modelB'],group[1].iloc[i]['modelC']]
        df_json.to_json(output_path + 'triplet_' + str(i) + '.json',orient='records',lines=True)


if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", default=project_path + 'data/interim/paraphrasing_eval_dataset/paraphrasing_evaluation_dataset.csv',type=str)
    argparser.add_argument("--output_path", default=project_path + 'data/interim/human_eval_paraphrasing_judgellm_round1/',type=str)
    data_path = argparser.parse_args().data_path
    output_path = argparser.parse_args().output_path
    df= pd.read_csv(data_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    generate_triplets_evaluation(df,output_path)
    
