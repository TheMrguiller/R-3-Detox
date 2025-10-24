import pandas as pd
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import hashlib


def get_complete_path_of_file(filename):
    """Join the path of the current directory with the input filename."""
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, filename)


def read_wordlist(filename: str):
    """Return words from a wordlist file."""
    with open(filename, encoding="utf-8") as wordlist_file:
        for row in iter(wordlist_file):
            row = row.strip()
            if row != "":
                yield row


def any_next_words_form_swear_word(cur_word, words_indices, censor_words):
    """
    Return True, and the end index of the word in the text,
    if any word formed in words_indices is in `CENSOR_WORDSET`.
    """
    # print("cur_word", cur_word)
    full_word = cur_word.lower()
    full_word_with_separators = cur_word.lower()

    # Check both words in the pairs
    for index in iter(range(0, len(words_indices), 2)):
        single_word, end_index = words_indices[index]
        word_with_separators, _ = words_indices[index + 1]
        if single_word == "":
            continue

        full_word = "%s%s" % (full_word, single_word.lower())
        full_word_with_separators = "%s%s" % (
            full_word_with_separators,
            word_with_separators.lower(),
        )
        # print("full_word", full_word)
        # print(censor_words)
        if full_word in censor_words or full_word_with_separators in censor_words or cur_word in censor_words:
            if full_word in censor_words:
                try:
                    
                    index_= censor_words.index(full_word)
                    # print(index_)
                    original_word = str(censor_words[index_])
                    return True, end_index,original_word
                except ValueError:
                    return False, -1,None
            elif cur_word in censor_words:
                try:
                    index_= censor_words.index(cur_word)
                    original_word = str(censor_words[index_])
                    return True, end_index,original_word
                except ValueError:
                    return False, -1,None
            elif full_word_with_separators in censor_words:
                try:
                    index_= censor_words.index(full_word_with_separators)
                    original_word = str(censor_words[index_])
                    return True, end_index,original_word
                except ValueError:
                    return False, -1,None
    return False, -1, None

def chunkify(df, num_chunks):
        chunk_size = len(df) // num_chunks
        return [df[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    
def process_chunk_wrapper(args):
    chunk, processing_func, column_text = args
    return processing_func(chunk, column_text)
    
def parallel_process_dataframe_with_progress_multi_function(df, num_processes,column_text, processing_func,desc="Processing rows"):
    with multiprocessing.Pool(processes=num_processes) as pool:
        chunks = chunkify(df, num_processes)
        with tqdm(total=len(chunks),desc=desc) as pbar:
            processed_chunks = []
            for result in pool.imap_unordered(process_chunk_wrapper, [(chunk, processing_func, column_text) for chunk in chunks]):
                processed_chunks.append(result)
                pbar.update(1)
    result_df = pd.concat(processed_chunks)
    result_df.reset_index(drop=True,inplace=True)
    return result_df

def generate_hash(text):
    """Generate an MD5 hash for the given text."""
    try:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    except Exception as e:
        raise Exception(f"Error hashing text: {text}")

def text_dedup_parallel(df, text_column)->pd.DataFrame:
    # Step 1: Generate hashes in parallel for each row's text
    df['text_hash'] = df[text_column].parallel_apply(generate_hash)
    
    # Step 2: Drop duplicates based on the generated hashes
    df_unique = df.drop_duplicates(subset='text_hash').drop(columns='text_hash')
    
    return df_unique