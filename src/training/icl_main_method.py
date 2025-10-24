#https://python.langchain.com/docs/how_to/few_shot_examples/#pass-the-examples-and-formatter-to-fewshotprompttemplate
import os
import sys
project_path=os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
import pandas as pd
import argparse
from src.utils.chains.paraphrase_generation import ParaphraseExperimentOffline
from typing import List

def get_experiment_results(experiment:ParaphraseExperimentOffline,sentences:List[str],paraphrases:List[str],labels:List[int],shap_values:List[List[str]],sources:List[str],indexes:List[int],save_path:str,experiment_type:str):

    results = experiment.run_experiment(sentences=sentences,labels=labels,shap_values=shap_values,sources=sources,indexes=indexes,batch_size=64)
    # print(results)
    df_try = pd.DataFrame(columns=["source","sentence","paraphrase","label","shap_values","result"])
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
    df_try.to_csv(save_path+f"experiment_{experiment_type}_results.csv",index=False)
    return True

def launch_experiments(df:pd.DataFrame,model_config:str,output_path:str):

    experiment = ParaphraseExperimentOffline(model_config=model_config,experiment_type="zero_shot",num_examples=0)
    sentences = df["sentence"].tolist()
    labels = df["label"].tolist()
    df["shap_values"]= df["shap_values"].apply(eval)
    shap_values = df["shap_values"].tolist()
    sources = df["source"].tolist()
    paraphrases = df["paraphrase"].tolist()
    indexes = df.index.tolist()

    for experiment_type in ["zero_shot","one_shot"]:#["zero_shot","one_shot","few_shot"]
        print(f"Experiment type: {experiment_type}")
        if experiment_type == "zero_shot":
            experiment.experiment_type = experiment_type
            experiment.num_examples = 0
            get_experiment_results(experiment=experiment,sentences=sentences,paraphrases=paraphrases,labels=labels,shap_values=shap_values,sources=sources,indexes=indexes,save_path=output_path,experiment_type=experiment_type)
        elif experiment_type == "one_shot":
            experiment.experiment_type = experiment_type
            experiment.num_examples = 1
            get_experiment_results(experiment=experiment,sentences=sentences,paraphrases=paraphrases,labels=labels,shap_values=shap_values,sources=sources,indexes=indexes,save_path=output_path,experiment_type=experiment_type+"_"+str(experiment.num_examples))

        elif experiment_type == "few_shot":
            experiment.experiment_type = experiment_type
            for num_examples in [5,7]:#[2,3,5,7,10]
                print(f"Number of examples: {num_examples}")
                experiment.num_examples = num_examples
                if not get_experiment_results(experiment=experiment,sentences=sentences,paraphrases=paraphrases,labels=labels,shap_values=shap_values,sources=sources,indexes=indexes,save_path=output_path,experiment_type=experiment_type+"_"+str(experiment.num_examples)):
                    break



if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name",type=str,default="marco-o1",help="Path to the model configuration file")
    argparser.add_argument("--data_path",type=str,default=project_path+"data/processed/final_few_shot_reasoning/few_shot_reasoning.csv",help="Path to the data file")
    #qwq_preview
    model_name = argparser.parse_args().model_name
    print(f"Model name: {model_name}")
    model_config =project_path+f"src/utils/llms/configs/{model_name}.yaml"
    data_path = argparser.parse_args().data_path
    df = pd.read_csv(data_path)
    df.drop(columns=["reasoning"],inplace=True)
    df = df[df["source"]!="non_toxic"]
    df.reset_index(drop=True,inplace=True)
    # df = df[:32]
    output_path = project_path+f"data/interim/paraphrasing_experiment/{model_name}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    launch_experiments(df=df,model_config=model_config,output_path=output_path)

    

    


    

    
