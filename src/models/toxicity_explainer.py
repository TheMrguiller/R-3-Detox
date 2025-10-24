import shap
import transformers
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import multiprocessing
from tqdm import tqdm
import re

class ToxicityExplainer:
    def __init__(self,model_name,batch_size=128):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_prediction_column_name= {
            "tomh/toxigen_hatebert":"LABEL_1",
            "tomh/toxigen_roberta":"LABEL_1",
            "unitary/toxic-bert":"toxic",
            "unitary/unbiased-toxic-roberta":"toxicity",
            "Xuhui/ToxDect-roberta-large":"LABEL_1"
        }
        model_to_tokenizer={
            "tomh/toxigen_hatebert":"GroNLP/hateBERT",
            "tomh/toxigen_roberta":"tomh/toxigen_roberta",
            "unitary/toxic-bert":"google-bert/bert-base-uncased",
            "unitary/unbiased-toxic-roberta":"unitary/unbiased-toxic-roberta",
            "Xuhui/ToxDect-roberta-large":"Xuhui/ToxDect-roberta-large"
        }
        self.model_name = model_name
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_to_tokenizer[model_name])
        self.pipeline = transformers.pipeline("text-classification", model=model_name,tokenizer=self.tokenizer,return_all_scores=True,device=device,batch_size=batch_size)
        # self.explainer = shap.Explainer(shap.models.TransformersPipeline(self.pipeline, rescale_to_logits=True))
        self.explainer = shap.Explainer(self.pipeline)
        label_config = self.pipeline.model.config.id2label

        self.prediction_column = model_prediction_column_name[model_name]
        self.index_to_label = [k for k, v in label_config.items() if v == self.prediction_column][0]
        self.num_workers = multiprocessing.cpu_count() if os.getenv("SLURM_CPUS_PER_TASK") is None else int(os.getenv("SLURM_CPUS_PER_TASK"))
        self.batch_size = batch_size
    def softmax_stable(self,x):
        return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

    def explain(self, texts: list):
        predictions = self.pipeline(texts)
        # print(predictions)
        batch_explanations=self.explainer(texts,batch_size=self.batch_size)
        # print(batch_explanations)
        batch_result=[]
        batch_explanations_shap_values=list(batch_explanations[:,:,self.prediction_column].values)
        batch_explanations_tokenized_text=list(batch_explanations[:,:,self.prediction_column].data)
        for idx,(explation_shap, explanation_tokenized_text) in tqdm(enumerate(zip(batch_explanations_shap_values,batch_explanations_tokenized_text))):
            result= {}
            # print(explanation)
            # print(type(explanation[:,self.prediction_column].values))
            result["shap_values"]=list(explation_shap)

            result["prediction"]=predictions[idx][self.index_to_label]["score"]
            # print(type(explanation[:,self.prediction_column].data))
            result["tokenized_text"]=list(explanation_tokenized_text)

            batch_result.append(result)
        return batch_result
    
    # def explain(self, text: list):
        
    #     # Convert text to DataFrame for batching
    #     text_df = pd.DataFrame(text, columns=["text"])

    #     # Define a function to process each batch
    #     def process_batch(batch_start, batch_end):
    #         pipeline = transformers.pipeline("text-classification", model=model_name,tokenizer=self.tokenizer,return_all_scores=True,device=device)
    #         batch_result = []
    #         batch = text_df.iloc[batch_start:batch_end]
    #         predictions = self.pipeline(batch["text"].tolist())
    #         batch_explanations = self.explainer(batch["text"].tolist())
            
    #         for idx, explanation in enumerate(batch_explanations):
    #             result = {}
    #             print(explanation)
    #             result["shap_values"] = explanation[:, self.prediction_column].values
    #             result["prediction"] = predictions[idx][self.index_to_label]["score"]
    #             result["tokenized_text"] = explanation[:, self.prediction_column].data
    #             batch_result.append(result)

    #         return batch_result

    #     explanations = []

    #     # Use ProcessPoolExecutor to parallelize the processing
    #     with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
    #         futures = []
    #         for start in range(0, len(text), self.batch_size):
    #             end = min(start + self.batch_size, len(text))
    #             futures.append(executor.submit(process_batch, start, end))

    #         for future in futures:
    #             explanations.append(future.result())

    #     # Concatenate results
    #     return np.concatenate(explanations, axis=0)
    



    