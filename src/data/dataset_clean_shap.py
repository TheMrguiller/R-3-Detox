import re
import pandas as pd
import ast
from string import punctuation
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer
import os
import multiprocessing
from pandarallel import pandarallel

num_cores=multiprocessing.cpu_count() if os.getenv("SLURM_CPUS_PER_TASK") is None else int(os.getenv("SLURM_CPUS_PER_TASK"))
pandarallel.initialize(progress_bar=True,nb_workers=num_cores)
project_path=os.path.abspath(__file__).split('src')[0]
nltk.download('stopwords')
stopword_en=set(stopwords.words('english'))
for word in["not", "no", "nor", "against", "never", "none", "nobody", "nothing", "nowhere", "neither", "without", "scarcely", "hardly", "barely"]:
  if word in stopword_en:
    stopword_en.remove(word)


punct = set(punctuation)

def tokenizer_with_positions(sentence,tokenizer):
  encoding = tokenizer(sentence, return_offsets_mapping=True,add_special_tokens=False)
  offsets = encoding['offset_mapping']
  tokens = tokenizer.tokenize(sentence)
  tokens_with_positions = []
  for token, offset in zip(tokens, offsets):
    tokens_with_positions.append((token, offset[0], offset[1]))
  return tokens_with_positions

def tokenize_with_positions(sentence):
    # Define the regex pattern for words and punctuation
    pattern = r"\b\w+(?:['-]\w+)*\b|[^\w\s]"

    # Use re.finditer to find all matches along with their positions
    tokens_with_positions = []
    for match in re.finditer(pattern, sentence):
        token = match.group()  # The actual token
        start_pos = match.start()  # Start position in the original text
        end_pos = match.end()  # End position in the original text
        tokens_with_positions.append((token, start_pos, end_pos))

    return tokens_with_positions

def zip_shap_values_to_tokens(shap_values,tokens):
  tokens_with_shap_values = []
  for shap_value,token in zip(shap_values,tokens):
    if token!="":
      tokens_with_shap_values.append((token,shap_value))
  return tokens_with_shap_values

def unifiy_shap_pos(pos_tokens,shap_tokens):
  unify_pos_shap = []
  for pos_token,shap_token in zip(pos_tokens,shap_tokens):
    unify_pos_shap.append((pos_token[0],pos_token[1],pos_token[2],shap_token[1]))
  return unify_pos_shap

def obtain_word_shap_value(word_tokens,shap_tokens):
  final_word_shap=[]
  index=0
  for idx,word_token in enumerate(word_tokens):
    word, start,end = word_token
    final_word_shap.append((word,[]))

    for shap_token in shap_tokens[index:]:
      if shap_token[1]>=start and shap_token[2]<=end:
        final_word_shap[idx][1].append(shap_token[3])
  return final_word_shap

def obtain_final_sentence_shap(tokenizer,sentence,shap_values,tokens):
  tokens_positions_simple = tokenize_with_positions(sentence)
  tokens_positions_tokenizer = tokenizer_with_positions(sentence,tokenizer)
  shap_tokens = zip_shap_values_to_tokens(shap_values,tokens)
  unify_pos_shap = unifiy_shap_pos(tokens_positions_tokenizer,shap_tokens)
  final_word_shap = obtain_word_shap_value(tokens_positions_simple,unify_pos_shap)
  max_word_shap= []
  for idx,word_shap in enumerate(final_word_shap):
    word,shap_values = word_shap
    if not non_relevant_words(word):
      if len(shap_values)>0:
        max_word_shap.append((word,max(shap_values)))
  return max_word_shap

def is_number(s):
    return bool(re.match(r'^-?\d+(\.\d+)?$', s))

def non_relevant_words(word:str):
  if word in stopword_en:
    return True
  if word in punct:
    return True
  if is_number(word):
    return True
  if word.lower() in ["i","you","he","she","it","we","they","us","them",
                      "my","your","his","her","its","our","their","mine","yours","his","hers","ours","theirs",
                      "me","him","her","us","them",
                      "i'm","you're","he's","she's","it's","we're","they're",
                      "i've","you've","he's","she's","it's","we've","they've",
                      "i'll","you'll","he'll","she'll","it'll","we'll","they'll",
                      "i'd","you'd","he'd","she'd","it'd","we'd","they'd",
                      ]:
    return True
  return False

def process_explanations(dataset_path:str,model_name:str)->pd.DataFrame:
    print(f"Processing model {model_name}")
    df = pd.read_csv(dataset_path)
    df["shap_values"] = df["shap_values"].parallel_apply(ast.literal_eval)
    df["tokenized_text"] = df["tokenized_text"].parallel_apply(ast.literal_eval)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df["shap_values"] = df.parallel_apply(lambda x: obtain_final_sentence_shap(tokenizer,x["sentence"],x["shap_values"],x["tokenized_text"]),axis=1)
    df.drop(columns=["tokenized_text"],inplace=True)
    if model_name=="google-bert/bert-base-uncased":
      model_name = "unitary_toxic-bert"
    elif model_name=="GroNLP/hateBERT":
      model_name = "tomh_toxigen_hatebert"
    model_name = model_name.replace("/","_")
    if not os.path.exists(project_path+f"data/processed/shap_values/"):
        os.makedirs(project_path+f"data/processed/shap_values/")
    
    df.to_csv(project_path+f"data/processed/shap_values/processed_{model_name}.csv",index=False)

if __name__ == '__main__':
    print("Processing dataset")
    process_explanations(project_path+"data/interim/shap_values/dataset_tomh_toxigen_hatebert.csv","GroNLP/hateBERT")
    process_explanations(project_path+"data/interim/shap_values/dataset_tomh_toxigen_roberta.csv","tomh/toxigen_roberta")
    process_explanations(project_path+"data/interim/shap_values/dataset_unitary_toxic-bert.csv","google-bert/bert-base-uncased")
    process_explanations(project_path+"data/interim/shap_values/dataset_unitary_unbiased-toxic-roberta.csv","unitary/unbiased-toxic-roberta")
    process_explanations(project_path+"data/interim/shap_values/dataset_Xuhui_ToxDect-roberta-large.csv","Xuhui/ToxDect-roberta-large")
    