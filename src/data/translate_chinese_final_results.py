import pandas as pd
import os 
import sys
project_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
import glob
from src.utils.translate import translate_text,detect_chinese_characters
import random
import argparse
from src.training.icl_main_method_with_error_fixing import row_dict_value_empty
from time import sleep
from tqdm import tqdm
tqdm.pandas()
def translate_with_sleep(value,column_name="paraphrase"):
    if detect_chinese_characters(value[column_name]):
        value[column_name]=translate_text(value[column_name])
        sleep_time = random.uniform(0.25,1.5)
        sleep(sleep_time)
    return value
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--folder",type=str,help="Folder we want to extract the paraphrase from",default="qwq_preview")#llama3_1_8B,openO1,marco-o1,qwen2_5_7B,qwq_preview
    folder = args.parse_args().folder
    source_file=project_path+"data/processed/final_paraphrases/"+folder+"/"
    output_path = project_path+"data/processed/final_paraphrases_with_language_error/"+folder+"/"
    
    files = glob.glob(source_file+"*.csv")
    for file in tqdm(files,desc="Files",total=len(files)):
        df = pd.read_csv(file)
        df["result"] = df["result"].apply(lambda x: eval(x))
        filename = file.split("/")[-1]
        df_with_hallucinations = df[(df["result"]=="Model hallucinated") | (df["result"]=="") | (df["result"].apply(row_dict_value_empty))]
        print(f"total hallucinations: {len(df_with_hallucinations)}, model: {filename}")
        number_of_chinese_elements_paraphrase = df["result"].apply(lambda x: 1 if detect_chinese_characters(x["paraphrase"]) else 0)
        number_of_chinese_elements_reasoning = df["result"].apply(lambda x: 1 if detect_chinese_characters(x["reasoning"]) else 0)
        number_of_chinese_elements_paraphrase = number_of_chinese_elements_paraphrase.sum()
        number_of_chinese_elements_reasoning = number_of_chinese_elements_reasoning.sum()
        print(f"Number of chinese elements in {filename} is paraphrase:{number_of_chinese_elements_paraphrase}, reasoning:{number_of_chinese_elements_reasoning}")
        if number_of_chinese_elements_paraphrase>0 or number_of_chinese_elements_reasoning>0:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            df.to_csv(output_path+filename,index=False)
            if number_of_chinese_elements_paraphrase>0:
                df["result"] = df["result"].progress_apply(lambda x: translate_with_sleep(x,"paraphrase"))
            if number_of_chinese_elements_reasoning>0:
                df["result"] = df["result"].progress_apply(lambda x: translate_with_sleep(x,"reasoning"))
            df.to_csv(file,index=False)