from bert_score import score
from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from transformers import AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os
import Levenshtein

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def bert_score(hypotheses, references):
    """
    Calculate the Bert Score for the hypotheses and references
    """
    P, R, F1 = score(cands=hypotheses, refs=references, lang='en', verbose=False,model_type="allenai/longformer-large-4096")
    mean_f1 = F1.mean().item()
    std_f1 = F1.std().item()
    min_f1 = F1.min().item()
    max_f1 = F1.max().item()
    return mean_f1, std_f1, min_f1, max_f1

def obtain_similarity_text_bert_score(texts:List[str])->float:
    """
    Calculate the Bert Score for model generated text from the same sentence
    """
    hypotheses = []
    references = []
    for i in range(len(texts)-1):
        for j in range(i+1,len(texts)):
            hypotheses.append(texts[i])
            references.append(texts[j:])
            break
    return bert_score(hypotheses, references)

def calculate_self_bleu_all_ngrams(generated_sentences):
    smooth_fn = SmoothingFunction().method1
    bleu_scores = {n: [] for n in range(1, 5)}  # For n-grams 1 to 4

    for i, hypothesis in enumerate(generated_sentences):
        references = [
            sentence.split() for j, sentence in enumerate(generated_sentences) if j != i
        ]
        hypothesis_tokens = hypothesis.split()

        for n in range(1, 5):  # Loop over 1-gram to 4-gram
            weights = tuple((1.0 / n if i < n else 0) for i in range(4))
            bleu_score = sentence_bleu(
                references,
                hypothesis_tokens,
                weights=weights,
                smoothing_function=smooth_fn
            )
            bleu_scores[n].append(bleu_score)

    # Average scores for each n-gram
    return {n: [np.mean(scores),np.std(scores),np.min(scores),np.max(scores)] for n, scores in bleu_scores.items()}

def semantic_similarity(texts:List[str])->float:
    """
    Calculate the semantic similarity between the texts
    """
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    embeddings = model.encode(texts, task="text-matching")
    # Calculate pairwise cosine similarity
    if type(embeddings)==np.ndarray:
        similarity_matrix = cosine_similarity(embeddings)
    else:
        similarity_matrix = cosine_similarity(embeddings.numpy())
    
    # Return the average similarity score
    num_pairs = len(texts) * (len(texts) - 1) / 2  # Number of unique pairs
    average_similarity = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].mean()
    std_similarity = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].std()
    min_similarity = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].min()
    max_similarity = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].max()
    return average_similarity, std_similarity, min_similarity, max_similarity

def levenshtein_distance(reference:str,candidates:List[str])->float:
    """
    Calculate the levenshtein distance between the texts
    """
    distances = []
    for candidate in candidates:
        distances.append(Levenshtein.distance(reference,candidate))
    return distances
    

     
if __name__ == "__main__":
    import os
    import sys
    project_path=os.path.abspath(__file__).split('src')[0]
    sys.path.append(project_path)
    from src.utils.translate import translate_text
    import pandas as pd

    # Load the dataset
    dataset_path = project_path+"data/interim/few_shot_reasoning/few_shot_reasoning_qwq_preview.csv"
    df = pd.read_csv(dataset_path)
    df = df[:24]
    metric_df = pd.DataFrame(columns=["idx","sentence","semantic_similarity","bert_score","self_bleu"])
    for i, row in df.iterrows():
        texts = []
        texts.append(row['reasoning_greedy'])
        for text in eval(row["reasoning_sampling"]):
            text = translate_text(text)
            texts.append(text)
        sentence = row["sentence"]
        metric_df.loc[len(metric_df)] = [i, sentence, semantic_similarity(texts), obtain_similarity_text_bert_score(texts), calculate_self_bleu_all_ngrams(texts)]
        print(f"Idx: {i}")
        # print(f"Question: {sentence}")
        # print(f"Semantic Similarity: {semantic_similarity(texts)}")
        # print(f"Bert Score: {obtain_similarity_text_bert_score(texts)}")
        # print(f"Self Bleu: {calculate_self_bleu_all_ngrams(texts)}")
        # print("\n\n")
    metric_df.to_csv(project_path+"data/interim/few_shot_reasoning/metrics_few_shot_reasoning.csv",index=False)
    


            
