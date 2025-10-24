import pandas as pd
import os
import glob
import argparse
project_path=os.path.abspath(__file__).split('src')[0]
import sys
sys.path.append(project_path)
from typing import List
import random
from tqdm import tqdm


def generate_ranking_per_evaluation(result:str,model_names:List[str]):
    final_result = {}
    result_base = result.split(" ")
    result= []
    for i in range(len(result_base)):
        result.append(float(result_base[i]))
    for i in range(len(result)):
        final_result[model_names[i]] = result[i]
    sorted_result = sorted(final_result.items(), key=lambda x: x, reverse=False)
    ranked_result = {}
    max_value = sorted_result[0][1]
    rank=3
    for i in range(len(sorted_result)):
        if sorted_result[i][1] == max_value:
            ranked_result[sorted_result[i][0]] = rank
        else:
            rank+=1
            max_value = sorted_result[i][1]
            ranked_result[sorted_result[i][0]] = rank
    return ranked_result

        
    

        
    
    
    

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_prediction_path", default=project_path + 'data/processed/human_eval_paraphrasing_judgellm_round_1/')
    args = parser.parse_args()
    data_dir = args.judge_prediction_path
    second_round_evaluation_path = project_path + 'data/interim/human_eval_paraphrasing_judgellm_round2/'
    ranking_result_path = project_path + 'data/interim/judgellm_paraphrasing_ranking/'
    models = ["params33"]#["params7","params13","params33"]
    for model in models:
        files = glob.glob(data_dir+model+"/*.json")
        df_ranking_first_round = pd.DataFrame(columns=["idx","result"])
        df_ranking_second_round = pd.DataFrame(columns=['question_id','question_body','answer1_body','answer2_body','answer3_body','answer1_model_id','answer2_model_id','answer3_model_id'])
        dfs = [pd.read_json(file, lines=True) for file in files]
        for idx in tqdm(range(len(dfs[0]))):
            round_one_result = []
            for i in range(len(files)):
                round_one_result.append(generate_ranking_per_evaluation(dfs[i].loc[idx]["pred_text"],[dfs[i].loc[idx]["answer1_model_id"],dfs[i].loc[idx]["answer2_model_id"],dfs[i].loc[idx]["answer3_model_id"]]))
            df_ranking_first_round.loc[len(df_ranking_first_round)] = [idx,round_one_result]
            winner_models = []
            for result in round_one_result:
                winners=[]
                for key in result.keys():
                    if result[key] == 3:
                        winners.append(key)
                winner=random.choice(winners)
                winner_models.append(winner)
            column_names = []
            for i,winner in enumerate(winner_models):
                row = dfs[i].loc[idx]
                column_name = row[row == winner].index.tolist()[0]
                column_names.append(column_name)
            df_ranking_second_round.loc[len(df_ranking_second_round)] = [dfs[0].loc[idx]["question_id"],dfs[0].loc[idx]["question_body"],dfs[0].loc[idx][column_names[0].split("_model_id")[0]+"_body"],dfs[1].loc[idx][column_names[0].split("_model_id")[0]+"_body"],dfs[2].loc[idx][column_names[0].split("_model_id")[0]+"_body"],dfs[0].loc[idx][column_names[0]],dfs[1].loc[idx][column_names[1]],dfs[2].loc[idx][column_names[2]]]
        if not os.path.exists(second_round_evaluation_path):
            os.makedirs(second_round_evaluation_path)
        if not os.path.exists(ranking_result_path+model):
            os.makedirs(ranking_result_path+model)
        df_ranking_first_round.to_json(ranking_result_path+model+"/first_round.json",orient='records',lines=True)
        df_ranking_second_round.to_json(second_round_evaluation_path+model+".json",orient='records',lines=True)
    
                
            
