from bs4 import BeautifulSoup
import unidecode
import re
from emot.emo_unicode import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
from flashtext import KeywordProcessor
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
from pandarallel import pandarallel
import os
import demoji
import logging
from textblob import Word

from concurrent.futures import ProcessPoolExecutor, as_completed
logging.basicConfig(
    level=logging.DEBUG,
)
logger = logging.getLogger(
    __name__
)
num_cores=mp.cpu_count() if os.getenv("SLURM_CPUS_PER_TASK") is None else int(os.getenv("SLURM_CPUS_PER_TASK"))
pandarallel.initialize(progress_bar=True,nb_workers=num_cores)


class TextToxicityCleaner:
    '''
    This class is going to be used for cleaning the text data:
    - Eliminate HTML Tags check
    - Eliminate https and http links check
    - Eliminate @usernames check
    - Eliminate emotes in string and no string check
    - Eliminate accented characters check
    - Eliminate repeated characters in words check
    - URL in the OSemEval_2019_-_Task_6_-_Identifying_and_Categorizing_Offensive_Language_in_Social_Media is the https so need to eliminate or have that kind of thing into account
    Do not clean benchmark and analyze datasts
    '''
    def __init__(self):
        self.data = None
        
        all_emoji_emoticons = {**EMOTICONS_EMO,**UNICODE_EMOJI_ALIAS, **UNICODE_EMOJI_ALIAS}
        self.all_emoji_emoticons = {k:v.replace(":","").replace("_"," ").strip() for k,v in all_emoji_emoticons.items()}
        self.kp_all_emoji_emoticons = KeywordProcessor()
        for k,v in self.all_emoji_emoticons.items():
            self.kp_all_emoji_emoticons.add_keyword(k, v)

    def process_keywords(self,):
        kp = KeywordProcessor()
        for k,v in self.all_emoji_emoticons.items():
            kp.add_keyword(k, v)
        return kp
    def remove_extra_whitespaces_func(self,text:str):
        '''
        Removes extra whitespaces from a string, if present
        
        Args:
            text (str): String to which the function is to be applied, string
        
        Returns:
            Clean string without extra whitespaces
        ''' 
        try:
            text = text.encode('utf-8')
            # Decode from Latin-1 and re-encode to UTF-8
            text = text.decode('utf')
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            text=re.sub(r'^\s*|\s\s*', ' ', text).strip()
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    
            # Fix spaces before punctuation (e.g., " ? " -> "?", " , " -> ",")
            text = re.sub(r'\s+([?.!,])', r'\1', text)
            
            # Fix spaces after punctuation if missing
            text = re.sub(r'([?.!,])(\w)', r'\1 \2', text)
            
            # Normalize contractions (e.g., "can 't" -> "can't", "i 'm" -> "I'm")
            contractions = {
                r"can 't": "can't",
                r"i 'm": "I'm",
                r"n 't": "n't",
                r" 've": "'ve",
                r" 're": "'re",
                r" 'll": "'ll",
                r" 'd": "'d",
                r" 's": "'s",
            }
            for pattern, replacement in contractions.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
            # Capitalize the first letter of each sentence
            text = '. '.join(sentence.strip().capitalize() for sentence in text.split('.'))
            text = text.strip()
        except:
            logger.debug(f"Error in removing extra whitespaces from the text: {text}")
        
        return text
        
    def removeHTMLTags(self,text):
        '''
        Function to remove the HTML Tags from a given text.
        
        Parameter:
        ---------
        text: str
            Text from which the HTML tags has to be removed.
        '''
        
        # Reference: 'Remove html tags using BeautifulSoup' - https://www.geeksforgeeks.org/remove-all-style-scripts-and-html-tags-using-beautifulsoup/
        
        # Create a BeautifulSoup object to parse the given html text content
        try:
            soup = BeautifulSoup(text, 'html.parser')
            
            # Remove the <style> and <script> tags from the html content because they contains the styling sheet and javascript
            # file references and won't give any meaningful context.
            for data in soup(['style', 'script']):
                
                # Remove tag
                data.decompose()
                
            # Return the html tag free content
            text=   ' '.join(soup.stripped_strings)
        except:
            logger.debug(f"Error in removing HTML tags from the text: {text}")
        return text

    
    def removeAccentedChars(self,text):
        '''
        Function to remove the accented characters from a given text.
        
        Parameter:
        ---------
        text: str
            Text from which the accented character has to be removed.
        '''
        
        # Reference: "remove accented characters python" - https://www.geeksforgeeks.org/how-to-remove-string-accents-using-python-3/
        
        # Remove accents
        try:
            text = unidecode.unidecode(text)
        except:
            logger.debug(f"Error in removing accented characters from the text: {text}")
        return text
    
    def removeIPLinkNum(self,text, ipAddress=True, hyperlink=True, numbers=False):
        '''
        Function to remove IP Address and Number from the given text.
        
        Parameter:
        ---------
        text: str
            Text from which IP Address and number(s) have to be removed.
        '''
        
        # Replace IP Address with empty string.
        # Reference: 'Remove IP Address Python' - https://www.geeksforgeeks.org/extract-ip-address-from-file-using-python/#:~:text=The%20regular%20expression%20for%20valid,%5C.)%7B
        try:
            if ipAddress == True:
                
                text = re.sub(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', '', text)
            
            # Remove hyperlinks
            # Reference: 'Regex for hperlinks Python' - https://www.geeksforgeeks.org/python-check-url-string/
            if hyperlink == True:
                
                text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
            
            # Remove numbers.
            if numbers == True:
                
                text = re.sub(r'[0-9]', '', text)
            
            # Remove the extra space if any.
            text = re.sub(r'[ ][ ]+', ' ', text)
        except:
            logger.debug(f"Error in removing IP, Links and Numbers from the text: {text}")
        
        return text

    def clean_repeated_characters(self,text):
        try:
            rx = re.compile(r'([^\W\d_])\1{2,}')
            text = re.sub(r'[^\W\d_]+', lambda x: Word(rx.sub(r'\1\1', x.group())).correct() if rx.search(x.group()) else x.group(), text)
        except:
            logger.debug(f"Error in removing repeated characters from the text: {text}")
        return text
    
    def clean_emotes(self,text):
        # Replace emoji and emoticons in the text
        try:
            text = self.kp_all_emoji_emoticons.replace_keywords(text)
        except:
            logger.debug(f"Error in removing emotes from the text: {text}")
        return text
    
    def clean_emotes_v2(self,text,kp):
        # Replace emoji and emoticons in the text
        try:
            text = kp.replace_keywords(text)
        except:
            logger.debug(f"Error in removing emotes from the text: {text}")
        return text
    def clean_emotes_v3(self,text,kp):
        # Replace emoji and emoticons in the text
        try:
            text = demoji.replace(text, "")
        except:
            logger.debug(f"Error in removing emotes from the text: {text}")
        return text

    def select_only_arroba_starting_words(self,word_list):
        return [word for word in word_list if word.startswith('@')]
        
    def eliminate_arroba_username(self,text,obfuscated_words=None):
        try:
            if obfuscated_words is not None:
                obfuscated_words = self.select_only_arroba_starting_words(obfuscated_words)
                pattern = re.compile(r'@\S+', re.IGNORECASE)
                possible_arroba_user=pattern.findall(text)
                for word in possible_arroba_user:
                    if word not in obfuscated_words:
                        text = text.replace(word, '@USER')
                
                return text
            else:
                return re.sub(r'@\S+', '@USER', text)
        except:
            logger.debug(f"Error in removing @Usernames from the text: {text}")
        return text
    
    def chunkify(self,df, num_chunks):
        chunk_size = len(df) // num_chunks
        return [df[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

    def parallel_process_dataframe_with_progress_multi_function(self,df, num_processes, processing_func,text_column='comment_text',profane_word_column=None,description="Processing"):
        chunks = self.chunkify(df, num_processes)
        with mp.Manager() as manager:
            # Shared counter to track progress
            progress = manager.Value('i', 0)
            
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                for chunk in chunks:
                    future = executor.submit(self.process_chunk_wrapper, (chunk, processing_func,text_column,profane_word_column))
                    futures.append(future)

                processed_chunks = []
                with tqdm(total=len(chunks), desc=description, leave=False) as pbar:
                    # Update the progress bar while tasks are processing
                    for future in as_completed(futures):
                        try:
                            result = future.result()  # This will re-raise any exception in the submitted task
                            processed_chunks.append(result)
                            # Increment progress in the shared variable
                            progress.value += 1
                            # Update the progress bar
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Error processing chunk: {e}")

        # Concatenate processed chunks into a single DataFrame
        logger.debug(f"Concatenating processed chunks")
        result_df = pd.concat(processed_chunks, ignore_index=True)
        logger.debug(f"Processed chunks concatenated")
        #result_df.sort_index(inplace=True)
        result_df.dropna(subset=[text_column],inplace=True)
        logger.debug(f"Processed chunks dropped")
        result_df.reset_index(drop=True,inplace=True)
        logger.debug(f"Processed chunks reset")
        return result_df

    def process_chunk_wrapper(self,args):
        chunk, processing_func,text_column,profane_word_column = args
        return self.process_chunk(chunk, processing_func,text_column,profane_word_column)
    

    
    def clean_all(self,df,text_column="text",profane_word_column=None):
        kp=None
        for idx,row in df.iterrows():
            text = row[text_column]
            # logger.debug("Removing IP, Links and Numbers")
            text = self.removeIPLinkNum(text)
            # logger.debug("Removing Accented Characters")
            text = self.removeAccentedChars(text)
            # logger.debug("Removing HTML Tags")
            text = self.removeHTMLTags(text)
            # logger.debug("Removing @Usernames")
            text = self.eliminate_arroba_username(text,row[profane_word_column]) if row[profane_word_column]!=None else self.eliminate_arroba_username(text)
            
            # logger.debug("Removing Repeated Characters")
            text = self.clean_repeated_characters(text)
            # logger.debug("Removing Emotes")
            text = self.clean_emotes_v3(text,kp)
            text = self.clean_emotes(text)
            # logger.debug("Removing Extra")
            text = self.remove_extra_whitespaces_func(text)
            df.at[idx,text_column] = text
        return df

    
    def process_chunk(self,chunk, processing_func,text_column,profane_word_column=None):
        return processing_func(chunk,text_column,profane_word_column)
    
    def clean(self,df, text_column,profane_word_column=None,sleep_time=8):
        '''
        Function to do a general cleaning.
        
        Parameter:
        ---------
        df: pd.DataFrame
            Dataframe containing the text data.
        text_column: str
            Column name containing the text data.
        sleep_time: int
            Time to sleep for not having an overload.
        '''
        df = self.parallel_process_dataframe_with_progress_multi_function(df, num_cores, self.clean_all,text_column=text_column,profane_word_column=profane_word_column,  description="Removing all")

        # df["text"]= df.parallel_apply(lambda row: self.clean_all_row(row),axis=1)
        # df["text"]= df.apply(lambda row: self.clean_all_row(row),axis=1)
        # df=self.clean_all_spark(df)
        logger.debug("Cleaning Done")
        return df



