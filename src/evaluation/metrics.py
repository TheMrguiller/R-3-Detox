import os
import sys
project_path=os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
from src.evaluation.toxicity import ToxicMetric
from bert_score import score
import evaluate
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class ReferenceExperimentMetric:
    "Metrics presented in the paper: Demonstrations Are All You Need: Advancing Offensive Content Paraphrasing using In-Context Learning"
    def __init__(self):
        self.toxic_metric = ToxicMetric()
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        # self.CIDEr = evaluate.load("Kamichanw/CIDEr")
        # self.cider = Cider()
        # self.tokenizer = PTBTokenizer()
    def evaluate(self, reference:str, candidate:str):
        """
        Calculate the base metric for the experiment
        """
        toxic_score = self.toxic_metric.pred([candidate])[0]
        P, R, F1 = score(refs=[reference],cands=[candidate],lang='en',model_type="princeton-nlp/sup-simcse-roberta-large")
        bert_score = F1.item()
        rouge_score = self.rouge.compute(predictions=[candidate], references=[reference])["rougeL"]
        bleu_score = self.bleu.compute(predictions=[candidate], references=[reference],smooth=True)["bleu"]
        
        # cider_score = self.CIDEr.compute(predictions=[candidate], references=[[reference]])["CIDEr"]
        return toxic_score, bert_score, rouge_score, bleu_score#, cider_score
    
    def evaluate_batch(self, reference_list: list, candidate_list: list):
        toxic_scores = self.toxic_metric.pred(candidate_list)
        P, R, F1 = score(
            refs=reference_list,
            cands=candidate_list,
            lang='en',
            model_type="princeton-nlp/sup-simcse-roberta-large"
        )
        bert_scores = F1.tolist()
        rouge_scores = [
            self.rouge.compute(predictions=[candidate], references=[reference])["rougeL"]
            for reference, candidate in zip(reference_list, candidate_list)
        ]
        bleu_scores = [
            self.bleu.compute(predictions=[candidate], references=[reference], smooth=True)["bleu"]
            for reference, candidate in zip(reference_list, candidate_list)
        ]
        results = [
            [toxic_scores[i], bert_scores[i], rouge_scores[i], bleu_scores[i]]
            for i in range(len(reference_list))
        ]
        return results

class ReferenceFreeExperimentMetric:
    "Metrics presented in the paper: DetoxLLM: A Framework for Detoxification with Explanations"
    def __init__(self):

        self.content_similarity = SentenceTransformer('sentence-transformers/LaBSE')
        self.fluency = pipeline("text-classification",model="cointegrated/roberta-large-cola-krishna2020",device_map="auto",return_all_scores=True,truncation=True)
        self.style_transfer_accuracy = pipeline("text-classification",model="s-nlp/roberta_toxicity_classifier",device_map="auto",return_all_scores=True,truncation=True)
        self.bleu = evaluate.load("bleu")

    def evaluate(self, original:str, paraphrased:str):
        """
        Calculate the base metric for the experiment.
        Joint score is not calculated as int for all the metrics:
         - Percentage of neutral sentences detected by the style transfer model
         - Percentage of fluent sentences detected by the fluency model
         - Average of the content similarity of all the sentences
        """
        P, R, F1 = score(refs=[original],cands=[paraphrased],lang='en',model_type="princeton-nlp/sup-simcse-roberta-large")
        bert_score = F1.item()
        embeddings = self.content_similarity.encode([original, paraphrased])
        content_similarity = self.content_similarity.similarity(embeddings[0], embeddings[1]).numpy().item()
        fluency = int((self.fluency(paraphrased)[0][0]["score"]>0.5)) # Acceptability
        style_transfer = int(self.style_transfer_accuracy(paraphrased)[0][0]["score"]>0.5) # Neutral score
        #joint_score = content_similarity * fluency * style_transfer
        bleu_score = self.bleu.compute(predictions=[original], references=[paraphrased],smooth=True)["bleu"]

        return bert_score,bleu_score, content_similarity, fluency, style_transfer, #joint_score

    def evaluate_batch(self, original_list: list, paraphrased_list: list):
        """
        Evaluate a batch of original and paraphrased text pairs efficiently.

        Args:
            original_list (list): List of original texts.
            paraphrased_list (list): List of paraphrased texts.

        Returns:
            list: A list of tuples containing metrics for each pair.
        """
        if len(original_list) != len(paraphrased_list):
            raise ValueError("The lengths of original_list and paraphrased_list must be the same.")

        # Encode all texts at once for content similarity
        embeddings_original = self.content_similarity.encode(original_list)
        embeddings_paraphrased = self.content_similarity.encode(paraphrased_list)
        content_similarities = [
            self.content_similarity.similarity(embeddings_original[i], embeddings_paraphrased[i]).numpy().item()
            for i in range(len(original_list))
        ]

        # Compute fluency scores for all paraphrased texts
        fluency_scores = self.fluency(paraphrased_list)
        fluency_scores = [int(score[0]["score"] > 0.5) for score in fluency_scores]
        
        
        # Compute style transfer scores for all paraphrased texts
        style_transfer_scores = self.style_transfer_accuracy(paraphrased_list)
        style_transfer_scores = [int(score[0]["score"] > 0.5) for score in style_transfer_scores]


        # Compute BLEU scores for all pairs
        bleu_scores = [
            self.bleu.compute(predictions=[paraphrased_list[i]], references=[original_list[i]], smooth=True)["bleu"]
            for i in range(len(original_list))
        ]

        # Compute BERT scores for all pairs
        P, R, F1 = score(
            refs=original_list,
            cands=paraphrased_list,
            lang='en',
            model_type="princeton-nlp/sup-simcse-roberta-large"
        )
        bert_scores = F1.tolist()

        # Combine results for each pair
        results = [
            [bert_scores[i], bleu_scores[i], content_similarities[i], fluency_scores[i], style_transfer_scores[i]]
            for i in range(len(original_list))
        ]

        return results


    def obtain_joint_score(self, content_similarity:List[float], fluency:List[int], style_transfer:List[int]):
        """
        Calculate the joint score for the experiment
        """
        joint_score = 0
        content_similarity = sum(content_similarity)/len(content_similarity)
        fluency = sum(fluency)/len(fluency)
        style_transfer = sum(style_transfer)/len(style_transfer)
        joint_score = content_similarity * fluency * style_transfer

        return joint_score

import pandas as pd
from tqdm import tqdm
if __name__ == "__main__":
    # metric = ReferenceExperimentMetric()
    # reference_list = ["I love you","I hate you"]
    # # candidate = "I fucking hate you so much useless piece of shit"
    # candidate_list = ["I love you","I love you"]
    # metric_referencefree = ReferenceFreeExperimentMetric()
    # # print(metric.evaluate(reference, candidate))
    # print(f"Reference metric")
    # for reference,candidate in zip(reference_list,candidate_list):
    #     print(metric_referencefree.evaluate(reference,candidate))
    # print(metric_referencefree.evaluate_batch(original_list=reference_list,paraphrased_list=candidate_list))

    # print(f"Reference Free metric")
    # for reference,candidate in zip(reference_list,candidate_list):
    #     print(metric.evaluate(reference, candidate))
    # print(metric.evaluate_batch(reference_list=reference_list,candidate_list=candidate_list))
    df = pd.read_csv(project_path+"data/processed/final_few_shot_reasoning/few_shot_reasoning.csv")
    df = df[df["source"]!="non_toxic"]
    df.dropna(subset=["paraphrase"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    metric_referencefree = ReferenceFreeExperimentMetric()
    toxic_metric=ToxicMetric()
    batch_size = 32
    df_reference_free = pd.DataFrame(columns=["source","bert_scores","bleu_scores","content_similarities","fluency_scores","style_transfer_scores","toxic_scores"])
    for i in tqdm(range(0, len(df), batch_size), desc="Reference Free metric", total=len(df)//batch_size):
        # print(i)
        sources = df["source"][i:i+batch_size].tolist()
        sentences = df["sentence"][i:i+batch_size].tolist()
        paraphrases = df["paraphrase"][i:i+batch_size].tolist()
        results_reference_free=metric_referencefree.evaluate_batch(original_list=sentences,paraphrased_list=paraphrases)
        toxic_scores = toxic_metric.pred(paraphrases)
        for result,source,toxic_score in zip(results_reference_free,sources,toxic_scores):
            df_reference_free.loc[len(df_reference_free)] = [source]+result+[toxic_score]
        # df.loc[i:i+batch_size-1, "toxicity"] = values
    df_reference_free["idx"] = df.index
    if not os.path.exists(project_path+"results/metrics/paraphrase_automatic_metrics/human/human/"):
        os.makedirs(project_path+"results/metrics/paraphrase_automatic_metrics/human/human/")
    df_reference_free.to_csv(project_path+"results/metrics/paraphrase_automatic_metrics/human/human/metrics.csv",index=False)
    sources = df["source"].unique().tolist()
    for source in sources:
        df_reference_free_source = df_reference_free[df_reference_free["source"]==source]
        reference_free_json_results = {
            "bert_scores":df_reference_free_source["bert_scores"].mean(),
            "bleu_scores":df_reference_free_source["bleu_scores"].mean(),
            "content_similarities":df_reference_free_source["content_similarities"].mean(),
            "fluency_scores":df_reference_free_source["fluency_scores"].mean(),
            "style_transfer_scores":df_reference_free_source["style_transfer_scores"].mean(),
            "joint_score":metric_referencefree.obtain_joint_score(content_similarity=df_reference_free_source["content_similarities"].tolist(),fluency=df_reference_free_source["fluency_scores"].tolist(),style_transfer=df_reference_free_source["style_transfer_scores"].tolist()),
            "toxic_scores":df_reference_free_source["toxic_scores"].mean()
        }
        with open(project_path+"results/reports/paraphrase_automatic_metrics/final/"+f"{source}_reference_free_metrics.json","w") as f:
            f.write(str(reference_free_json_results))