import pandas as pd
import sys
import os
project_path=os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
from src.evaluation.prediction_quality import entropy_metric, log_loss_metric, mutual_info_metric,accuracy_metric
import numpy as np
from pprint import pprint
from pandarallel import pandarallel
import multiprocessing
import ast
num_cores=multiprocessing.cpu_count() if os.getenv("SLURM_CPUS_PER_TASK") is None else int(os.getenv("SLURM_CPUS_PER_TASK"))
pandarallel.initialize(progress_bar=True,nb_workers=num_cores)
project_path=os.path.abspath(__file__).split('src')[0]

def calculate_model_prediction_quality(prediction,label):

    prediction_array = np.array([[1-prediction, prediction] for prediction in prediction])
    prediction_integers = np.array([1 if pred>0.5 else 0 for pred in prediction])
    label = np.array(label)
    confidence_toxic=np.mean(np.array(prediction)[label==1])
    confidence_non_toxic=np.mean(np.array(prediction)[label==0])
    entropy_quality = entropy_metric(prediction_array)
    log_loss_quality = log_loss_metric(prediction_array,label)
    mutual_info_quality = mutual_info_metric(prediction_integers,label)
    accuracy_quality = accuracy_metric(prediction_integers,label)
    
    return entropy_quality,log_loss_quality,mutual_info_quality,accuracy_quality,confidence_toxic,confidence_non_toxic

def is_accurate_prediction(prediction,label):
    prediction = int(prediction > 0.5)
    return prediction == label

def compare_predictions_between_dataset(predictions, labels):
    predictions=np.array(predictions)
    predictions = np.transpose(predictions)
    
    prediction_int =[]
    for prediction in predictions:
        
        parcial_pred = []
        for pred in prediction:
            if pred >= 0.7:
                # print(pred)
                parcial_pred.append(1)
            else:
                parcial_pred.append(0)
        prediction_int.append(parcial_pred)


    
    result = []
    for row, label in zip(prediction_int,labels):
        # print(row,label)
        row_result = []
        for prediction in row:
            if prediction == label:
                row_result.append(1)
            else:
                row_result.append(0)
        print(row_result,label)
        result.append(row_result)
    final_result = [1 if sum(row)==0 else 0 for row in result]
    
    print(f"No accurate predictions in:{sum(final_result)}")
    return final_result

def get_common_shap_values(shap_values:list):
    common_shap_values = []
    shap_values_overall = [shap_value[0] for shap_value_list in shap_values for shap_value in shap_value_list]
    shap_values_processed = [[shap_value[0] for shap_value in shap_value_list] for shap_value_list in shap_values]
    shap_values_overall = list(set(shap_values_overall))
    for shap_value in shap_values_overall:
        if all([shap_value in shap_value_list for shap_value_list in shap_values_processed]):
            common_shap_values.append(shap_value)

    return common_shap_values
def aggregate_shap_values(df_hatebert,df_hate_roberta,df_unitarY_bert,df_unitary_roberta,df_xuhui):
    for df in [df_hatebert,df_hate_roberta,df_unitarY_bert,df_unitary_roberta,df_xuhui]:
        df["shap_values"] = df["shap_values"].parallel_apply(lambda x: ast.literal_eval(x))
        df["shap_values"] = df["shap_values"].parallel_apply(lambda shap_values: [shap_value for shap_value in shap_values if shap_value[1] > 0])
        df["prediction"] = df["prediction"].parallel_apply(lambda x: 1.0 if x>=0.7 else 0.0)
    df_shap_values_aggregated = pd.DataFrame({
        "sentence":df_hatebert["sentence"],
        "paraphrase":df_hatebert["paraphrase"],
        "label":df_hatebert["label"],
        "source":df_hatebert["source"],
        "split":df_hatebert["split"],
        "prediction_hatebert":df_hatebert["prediction"],
        "prediction_hate_roberta":df_hate_roberta["prediction"],
        "prediction_unitary_bert":df_unitarY_bert["prediction"],
        "prediction_unitary_roberta":df_unitary_roberta["prediction"],
        "prediction_xuhui":df_xuhui["prediction"],
        "shap_values_hatebert":df_hatebert["shap_values"],
        "shap_values_hate_roberta":df_hate_roberta["shap_values"],
        "shap_values_unitary_bert":df_unitarY_bert["shap_values"],
        "shap_values_unitary_roberta":df_unitary_roberta["shap_values"],
        "shap_values_xuhui":df_xuhui["shap_values"],
    })
    prediction_columns = ["prediction_hatebert","prediction_hate_roberta","prediction_unitary_bert","prediction_unitary_roberta","prediction_xuhui"]
    shape_columns = ["shap_values_hatebert","shap_values_hate_roberta","shap_values_unitary_bert","shap_values_unitary_roberta","shap_values_xuhui"]
    df_shap_values_aggregated["shap_values_aggregated"] = df_shap_values_aggregated.apply(
    lambda x: [
        x[shap_column]
        for prediction_column, shap_column in zip(prediction_columns, shape_columns)
        if x[prediction_column] == x["label"]
    ],
    axis=1
    )
    if len(df_shap_values_aggregated[df_shap_values_aggregated["shap_values_aggregated"].apply(len)==0])>0:
        df_shap_values_aggregated["shap_values_aggregated"] = df_shap_values_aggregated.apply(
            lambda x: 
            [x["shap_values_hatebert"], x["shap_values_hate_roberta"], x["shap_values_unitary_bert"], x["shap_values_unitary_roberta"], x["shap_values_xuhui"]]
            if x["prediction_hatebert"] != x["label"] 
            and x["prediction_hate_roberta"] != x["label"] 
            and x["prediction_unitary_bert"] != x["label"] 
            and x["prediction_unitary_roberta"] != x["label"] 
            and x["prediction_xuhui"] != x["label"] 
            else x["shap_values_aggregated"],
            axis=1
        )
        # df_shap_values_aggregated["prediction_aggregated"] = df_shap_values_aggregated.apply(
        #     lambda x: "Incorrect" 
        #     if x["prediction_hatebert"] != x["label"] 
        #     and x["prediction_hate_roberta"] != x["label"] 
        #     and x["prediction_unitary_bert"] != x["label"] 
        #     and x["prediction_unitary_roberta"] != x["label"] 
        #     and x["prediction_xuhui"] != x["label"] 
        #     else "Correct",
        #     axis=1
        # )
    df_shap_values_aggregated["shap_values_aggregated"] = df_shap_values_aggregated["shap_values_aggregated"].apply(lambda x: 
        get_common_shap_values(x))
    df_shap_values_aggregated.drop(columns=prediction_columns+shape_columns,inplace=True)
    df_shap_values_aggregated.rename(columns={"shap_values_aggregated":"shap_values"},inplace=True)
    return df_shap_values_aggregated


if __name__ == "__main__":
    # Datasets of the shap values
    df_hatebert = pd.read_csv(project_path+"data/processed/shap_values/processed_tomh_toxigen_hatebert_left_no_toxic.csv")
    df_hate_roberta = pd.read_csv(project_path+"data/processed/shap_values/processed_tomh_toxigen_roberta_left_no_toxic.csv")
    df_unitarY_bert = pd.read_csv(project_path+"data/processed/shap_values/processed_unitary_toxic-bert_left_no_toxic.csv")
    df_unitary_roberta = pd.read_csv(project_path+"data/processed/shap_values/processed_unitary_unbiased-toxic-roberta_left_no_toxic.csv")
    df_xuhui = pd.read_csv(project_path+"data/processed/shap_values/processed_Xuhui_ToxDect-roberta-large_left_no_toxic.csv")
    
    ## Calculate the quality of the predictions of the models
    # model_metrics = {}
    # for model_name,df in zip(["toxigen_hatebert","unitary_toxic-bert","unitary_unbiased-toxic-roberta","Xuhui_ToxDect-roberta-large","tomh_toxigen_roberta"],[df_hatebert,df_unitarY_bert,df_unitary_roberta,df_xuhui,df_hate_roberta]):
    #     entropy_quality,log_loss_quality,mutual_info_quality,accuracy_quality,confidence_toxic,confidence_non_toxic = calculate_model_prediction_quality(df["prediction"].to_list(),df["label"].to_list())
    #     df["accurate_prediction"] = df.apply(lambda x: is_accurate_prediction(x["prediction"],x["label"]),axis=1)
    #     df["entropy_quality"] = log_loss_quality
    #     model_metrics[model_name] = {"entropy":np.mean(entropy_quality),"log_loss":log_loss_quality,"mutual_info":mutual_info_quality,"accuracy":accuracy_quality,"confidence_toxic":confidence_toxic,"confidence_non_toxic":confidence_non_toxic}
    # pprint(model_metrics)
    ## Compare the predictions between the models to get predictions that are incorrect across all models
    for df in [df_hatebert,df_unitarY_bert,df_unitary_roberta,df_xuhui,df_hate_roberta]:
        df["pred_label"]=df["prediction"].apply(lambda x: 1 if x>=0.7 else 0)
        incorrect=df[df["pred_label"]!=df["label"]]
        print(len(incorrect))
        print(incorrect[["sentence","prediction","label"]])
    index_hatebert = df_hatebert[df_hatebert["pred_label"]!=df_hatebert["label"]].index
    index_bert = df_unitarY_bert[df_unitarY_bert["pred_label"]!=df_unitarY_bert["label"]].index
    index_roberta = df_unitary_roberta[df_unitary_roberta["pred_label"]!=df_unitary_roberta["label"]].index
    index_xuhui = df_xuhui[df_xuhui["pred_label"]!=df_xuhui["label"]].index
    index_hate_roberta = df_hate_roberta[df_hate_roberta["pred_label"]!=df_hate_roberta["label"]].index
    common_indices = index_hatebert.intersection(index_bert).intersection(index_roberta).intersection(index_xuhui).intersection(index_hate_roberta)
    print(df_hatebert.loc[common_indices]["sentence"],df_hatebert.loc[common_indices]["prediction"],df_hatebert.loc[common_indices]["label"])
    common_toxic_df = df_hatebert.loc[common_indices][df_hatebert.loc[common_indices]["label"]==1]
    common_non_toxic_df = df_hatebert.loc[common_indices][df_hatebert.loc[common_indices]["label"]==0]
    print(f"Common toxic indices:{len(common_toxic_df)}, Common non toxic indices:{len(common_non_toxic_df)}")


    ## Obtain the final aggregated shap values
    df_shap_values_aggregated = aggregate_shap_values(df_hatebert,df_hate_roberta,df_unitarY_bert,df_unitary_roberta,df_xuhui)
    if not os.path.exists(project_path+"data/processed/shap_values_aggregated/"):
        os.makedirs(project_path+"data/processed/shap_values_aggregated/")
    df_shap_values_aggregated.to_csv(project_path+"data/processed/shap_values_aggregated/processed_shap_values_left_no_toxic.csv",index=False)

