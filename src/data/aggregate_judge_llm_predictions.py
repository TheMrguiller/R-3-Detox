import pandas as pd
import os
import glob
import argparse
project_path=os.path.abspath(__file__).split('src')[0]

def find_reversed_pairs(data_dir):
    """
    Find pairs of directories where one has '_reversed' in its name and the other doesn't.

    Parameters:
        data_dir (str): The path to the directory containing subfolders.

    Returns:
        list: A list of tuples representing the pairs of directories.
    """
    # Get all subfolders under the given path
    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

    # Find pairs of directories (one with '_reversed' and one without)
    pairs = []
    visited = set()
    for folder in subfolders:
        if folder.endswith("_reversed"):
            base_name = folder.replace("_reversed", "")
            if base_name in subfolders and base_name not in visited and folder not in visited:
                pairs.append((base_name, folder))
                visited.add(base_name)
                visited.add(folder)
        elif folder not in visited:
            reversed_name = folder + "_reversed"
            if reversed_name in subfolders and reversed_name not in visited:
                pairs.append((folder, reversed_name))
                visited.add(folder)
                visited.add(reversed_name)

    return pairs

def obtain_best_models(df):
    """
    Obtain the best models based on the majority vote from the human evaluation.

    Parameters:
        df (pd.DataFrame): A DataFrame containing the human evaluation results.

    Returns:
        dict: A dictionary containing the best models for each prompt.
    """
    best_models = {}
    for index, row in df.iterrows():
        if row["result"] == 0:
            if row["modelA"] not in best_models:
                best_models[row["modelA"]] = 0
            else:
                best_models[row["modelA"]] += 0
            if row["modelB"] not in best_models:
                best_models[row["modelB"]] = 0
            else:
                best_models[row["modelB"]] += 0
        elif row["result"] == 1:
            if row["modelA"] not in best_models:
                best_models[row["modelA"]] = 1
            else:
                best_models[row["modelA"]] += 1
        elif row["result"] == 2:
            if row["modelB"] not in best_models:
                best_models[row["modelB"]] = 1
            else:
                best_models[row["modelB"]] += 1
    model_list=set(df["modelA"].tolist()+df["modelB"].tolist())
    for model in model_list:
        if model not in best_models:
            best_models[model]=0
    return best_models

def aggregate_judge_predictions(data_dir, pairs,output_path):
    """
    Aggregate the predictions from the judge model for the given pairs of directories.

    Parameters:
        data_dir (str): The path to the directory containing subfolders.
        pairs (list): A list of tuples representing the pairs of directories.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated predictions.
    """
    for base_name, reversed_name in pairs:
        base_prediction_path = glob.glob(data_dir + base_name + "/*.json")
        reversed_prediction_path = glob.glob(data_dir + reversed_name + "/*.json")
        df_base = pd.DataFrame(columns=["idx","modelA","modelB","result"])
        for base_file, reversed_file in zip(base_prediction_path, reversed_prediction_path):
            base_df = pd.read_json(base_file, lines=True)
            reversed_df = pd.read_json(reversed_file,lines=True)
            for index, row in base_df.iterrows():
                result_base = row["pred_text"].split(" ")
                result_reversed = reversed_df.loc[index]["pred_text"].split(" ")
                result_reversed = result_reversed[::-1]
                result = [(float(result_base[i])+float(result_reversed[i]))/2 for i in range(len(result_base))]
                if result[0] > result[1]:
                    result = 1
                elif result[0] < result[1]:
                    result = 2
                elif result[0] == result[1]:
                    result = 0
                df_base.loc[len(df_base)] = [row["question_id"],row["answer1_model_id"],row["answer2_model_id"],result]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        best_models = obtain_best_models(df_base)
        print(f"Best models for {base_name}: {best_models}")
        open(output_path+base_name+'_best_models.txt', 'w').write(str(best_models))
        df_base.to_csv(output_path+base_name+'_aggregated.csv',index=False)
    return df_base
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_prediction_path", default=project_path + 'data/processed/judgellm_humman_eval_pairs_reasoning/')
    parser.add_argument("--output_path", default=project_path + 'data/processed/judgellm_pairs_humman_eval_reasoning_aggregated/')
    args = parser.parse_args()
    data_dir = args.judge_prediction_path
    output_path = args.output_path
    subfolder = os.path.basename(os.path.normpath(data_dir))  # Get the last part of the path
    pairs_path = find_reversed_pairs(data_dir)
    aggregate_judge_predictions(data_dir, pairs_path,output_path)
    

    