import sys
import os
project_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
import outlines
import outlines.generate
import outlines.generate.json
import yaml
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from src.utils.structured_output.paraphrase_generation import ParaphraseGeneration
from src.utils.llms.llm_vllm_server_proxy import LLMVLLMProxy
from src.utils.llms.llm_huggingface_proxy import LLMHuggingFace
from src.utils.llms.llm_vllm_offline_proxy import LLMVLLMOfflineProxy
from time import sleep
import random
# from lmformatenforcer import JsonSchemaParser
# from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
# from transformers import pipeline

# from collections import Counter
# from math import log2
# import nltk

# # Download nltk tokenizer if not already available
# nltk.download('punkt')

# def n_gram_analysis(text, n=4):
#     """
#     Perform n-gram analysis to compute repetition ratios.

#     Args:
#         text (str): The input text.
#         n (int): The size of n-grams (e.g., 2 for bigrams, 3 for trigrams).

#     Returns:
#         dict: A dictionary with total n-grams, unique n-grams, and repetition ratio.
#     """
#     # Tokenize the text into words
#     words = nltk.word_tokenize(text)

#     # Generate n-grams
#     n_grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

#     # Count n-grams
#     n_gram_counts = Counter(n_grams)

#     # Calculate metrics
#     total_n_grams = len(n_grams)
#     unique_n_grams = len(n_gram_counts)
#     repetition_ratio = unique_n_grams / total_n_grams if total_n_grams > 0 else 0

#     return {
#         "total_n_grams": total_n_grams,
#         "unique_n_grams": unique_n_grams,
#         "repetition_ratio": repetition_ratio,
#         "most_common_n_grams": n_gram_counts.most_common(5)  # Top 5 most common n-grams
#     }

# def calculate_entropy(text):
#     """
#     Calculate Shannon entropy of the text based on character frequencies.

#     Args:
#         text (str): The input text.

#     Returns:
#         float: The Shannon entropy of the text.
#     """
#     # Count character frequencies
#     char_counts = Counter(text)
#     total_chars = len(text)

#     # Calculate entropy
#     entropy = -sum((count / total_chars) * log2(count / total_chars) for count in char_counts.values())

#     return entropy
"https://github.com/google-research/pegasus/tree/main"
class StructuredOutputEnforcer:
    def __init__(self, model_config:str):
        # self.llm = LLMHuggingFace(model_config)
        self.model_config = model_config
        self.llm = LLMVLLMProxy(model_config)
        self.llm_sampler = LLMVLLMProxy(self.model_config,temperature=0.7)
        # self.llm = LLMVLLMOfflineProxy(model_config)
        self.system_prompt = """You are a JSON extractor. You are given a text, and you need to extract the **reasoning** and the **final paraphrase**. Follow these guidelines:
1. The keys and values must be enclosed in double quotes (").
2. Strings inside the values must be enclosed in single quotes (').
3. Only include the **final paraphrase** under the corresponding key.
4. The **reasoning** must be as complete as possible, including:
   - Analysis of any toxic words in the original text.
   - An overall explanation of why the text is considered toxic.
   - A detailed description of the changes made to the text to generate the final paraphrase.
"""
        self.parser = 'The output should be formatted as a JSON instance that conforms to the JSON schema below.Here is the output schema:\n```json\n{"reasoning": "The reasoning process generated", "paraphrase": "The final paraphrase generated"}\n```'
#         self.examples="""Here you have some examples of the input and output of the model:
# Example 1:
# Input:"```\n\n  final_reasoning: To address the provided sentence and understand its toxicity, let's break down the elements that contribute to its negative tone and suggest improvements for a more constructive communication approach.\nAnalysis of the Provided Sentence\nThe sentence in question is: 'I also expect people like you to run like intellectually bankrupt coward when confronted with your staggering ignorant hypocrisy.'\nIdentifying Toxic Elements\n\n\n'Intellectually bankrupt': This term suggests a lack of knowledge or understanding, which can be dismissive and may discourage further contribution from the individual.\n\n\n'Ignorant': This word is particularly problematic as it directly attacks the person's knowledge or intelligence, implying that they are unaware or uninformed in a derogatory manner.\n\n\n'Run': The use of 'run' in this context implies a lack of courage or unwillingness to engage in a discussion, which can be seen as a personal attack.\n\n\n'Hypocrisy': Calling someone out for hypocrisy can be seen as a valid critique, but the way it's phrased here can come across as accusatory and judgmental.\n\n\n'Like': The use of 'like you' personalizes the statement, making it more accusatory and potentially confrontational.\n\n\nWhy the Sentence is Toxic\nThe combination of these terms creates a harsh and condescending tone. Such language can instill defensiveness in the recipient and derail any productive dialogue, thereby qualifying the sentence as toxic.\nTransforming the Sentence\nTo transform this sentence into a non-toxic paraphrase, consider the following adjustments:\n\n\nFocus on the Idea, Not the Person: Instead of attacking the person's intelligence or the predictability of their comment, address the specific points they have made.\n\n\nUse Neutral Language: Opt for words that are less emotionally charged and more objective in their description.\n\n\nProvide Constructive Feedback: Offer insights or corrections in a manner that encourages understanding and improvement.\n\n\nSuggested Paraphrase\nA revised version could be: 'When confronted with your actions that seem to contradict your beliefs, I expect a more thoughtful and informed response from individuals who are aware of their intellectual responsibilities.'\nExplanation of Changes\n\n\nShift from Personal Attack to General Observation: By saying 'When confronted with your actions that seem to contradict your beliefs,' it acknowledges the behavior without directly attacking the person's character.\n\n\nEncouragement for Thoughtfulness: The phrase 'I expect a more thoughtful and informed response' shifts the focus to the desired behavior rather than criticizing the individual.\n\n\nEmphasis on Intellectual Responsibility: The addition of 'from individuals who are aware of their intellectual responsibilities' emphasizes the importance of personal growth and self-awareness, which is a more constructive approach.\n\n\nThis approach maintains the intention to point out inconsistencies in behavior but does so in a manner that is less confrontational and more respectful.\n```"
# Output:```json\n{"reasoning": "To address the provided sentence and understand its toxicity, let's break down the elements that contribute to its negative tone and suggest improvements for a more constructive communication approach.\nAnalysis of the Provided Sentence\nThe sentence in question is: 'I also expect people like you to run like intellectually bankrupt coward when confronted with your staggering ignorant hypocrisy.'\nIdentifying Toxic Elements\n\n\n'Intellectually bankrupt': This term suggests a lack of knowledge or understanding, which can be dismissive and may discourage further contribution from the individual.\n\n\n'Ignorant': This word is particularly problematic as it directly attacks the person's knowledge or intelligence, implying that they are unaware or uninformed in a derogatory manner.\n\n\n'Run': The use of 'run' in this context implies a lack of courage or unwillingness to engage in a discussion, which can be seen as a personal attack.\n\n\n'Hypocrisy': Calling someone out for hypocrisy can be seen as a valid critique, but the way it's phrased here can come across as accusatory and judgmental.\n\n\n'Like': The use of 'like you' personalizes the statement, making it more accusatory and potentially confrontational.\n\n\nWhy the Sentence is Toxic\nThe combination of these terms creates a harsh and condescending tone. Such language can instill defensiveness in the recipient and derail any productive dialogue, thereby qualifying the sentence as toxic.", "paraphrase": "When confronted with your actions that seem to contradict your beliefs, I expect a more thoughtful and informed response from individuals who are aware of their intellectual responsibilities."}\n```
# Example 2:
# Input:"```\n\n  final_reasoning: To address the toxicity in the provided sentence and generate a non-toxic paraphrase, follow these steps:\n\nIdentify Toxic Elements:\nExamine the sentence: 'What's up with this Twitter stuff? This stuff is so off sometimes! lol'\n\nRecognize that 'stuff' is a placeholder for 'shit' and 'off' is a mild euphemism for 'wack'. These terms can be considered informal and potentially offensive in some contexts.\n\n\nAnalyze Toxicity:\n\n\nUnderstand that the use of informal language and euphemisms for profanity can be perceived as disrespectful or inappropriate in formal or professional settings.\n\n\nGenerate Paraphrase:\n\nReplace the informal terms with more formal alternatives.\nChoose 'material' for 'stuff' to maintain the context of discussing Twitter content.\n\nSelect 'unappealing' for 'off' to convey the same meaning without using slang.\n\n\nEnsure the paraphrase is grammatically correct and maintains the original meaning: 'What's up with this Twitter material? This material is so unappealing sometimes! lol'\n\n\nVerify Changes:\n\nConfirm that the paraphrase removes the informal language and euphemisms, making the sentence more appropriate for a broader audience.\nEnsure the meaning remains intact and the sentence is now non-toxic.\n\nBy following these steps, we will effectively transform a toxic sentence into a non-toxic paraphrase while preserving the original intent.\n```"
# Output:```json\n{"reasoning": "To address the toxicity in the provided sentence and generate a non-toxic paraphrase, follow these steps:\n\nIdentify Toxic Elements:\nExamine the sentence: 'What's up with this Twitter stuff? This stuff is so off sometimes! lol'\n\nRecognize that 'stuff' is a placeholder for 'shit' and 'off' is a mild euphemism for 'wack. These terms can be considered informal and potentially offensive in some contexts.\n\n\nAnalyze Toxicity:\n\n\nUnderstand that the use of informal language and euphemisms for profanity can be perceived as disrespectful or inappropriate in formal or professional settings.", "paraphrase": "What's up with this Twitter material? This material is so unappealing sometimes! lol"}\n```
# Example 3:
# Input:"```\n\n  final_reasoning: To address the provided sentence and create a reasoning process for generating a non-toxic paraphrase, follow these steps:\n\nAnalyze Toxic Elements:\nExamine the sentence: 'i woulda loved rebecca black if she sang a song that did something besides state the fucking obvious.'\nThe words that struck as toxic are the following: 'fucking' and 'woulda'.\n\nAssess why these elements are toxic:\n\n'fucking' is a profanity that is widely considered offensive and inappropriate in polite conversation.\n'woulda' is a colloquial contraction for 'would have', which is less formal and potentially less clear in context.\n\n\n\nExplain Toxicity:\n\nThe use of 'fucking' to describe the content of Rebecca Black's songs is disrespectful and dismissive.\n\nThe informal contraction 'woulda' makes the sentence less clear and potentially more offensive.\n\n\nGenerate Paraphrase:\n\nModify the sentence to remove toxicity while maintaining its original intent:\nReplace 'fucking' with a neutral term like 'clearly'.\nReplace 'woulda' with the full form 'would have' for clarity and formality.\n\n\nResulting paraphrase: 'I would have loved Rebecca Black if she sang a song that did something besides state the clearly obvious.'\n\nBy following these steps, the sentence will be transformed into a non-toxic version that retains its original intent without resorting to disrespectful language or informal contractions.\n```"
# Output:```json\n{"reasoning": "To address the provided sentence and create a reasoning process for generating a non-toxic paraphrase, follow these steps:\n\nAnalyze Toxic Elements:\nExamine the sentence: 'i woulda loved rebecca black if she sang a song that did something besides state the fucking obvious.'\nThe words that struck as toxic are the following: 'fucking' and 'woulda.\n\nAssess why these elements are toxic:\n\n'fucking' is a profanity that is widely considered offensive and inappropriate in polite conversation.\n'woulda' is a colloquial contraction for 'would have', which is less formal and potentially less clear in context.\n\n\n\nExplain Toxicity:\n\nThe use of 'fucking' to describe the content of Rebecca Black's songs is disrespectful and dismissive.\n\nThe informal contraction 'woulda' makes the sentence less clear and potentially more offensive.", "paraphrase": "I would have loved Rebecca Black if she sang a song that did something besides state the clearly obvious."}\n``` 
# """
        self.examples="""Here you have some examples of the input and output of the model:
Example 1:
Input:"```\n\n  final_reasoning: To address the provided sentence and understand its toxicity, let's break down the elements that contribute to its negative tone and suggest improvements for a more constructive communication approach.\nAnalysis of the Provided Sentence\nThe sentence in question is: 'I also expect people like you to run like intellectually bankrupt coward when confronted with your staggering ignorant hypocrisy.'\nIdentifying Toxic Elements\n\n\n'Intellectually bankrupt': This term suggests a lack of knowledge or understanding, which can be dismissive and may discourage further contribution from the individual.\n\n\n'Ignorant': This word is particularly problematic as it directly attacks the person's knowledge or intelligence, implying that they are unaware or uninformed in a derogatory manner.\n\n\n'Run': The use of 'run' in this context implies a lack of courage or unwillingness to engage in a discussion, which can be seen as a personal attack.\n\n\n'Hypocrisy': Calling someone out for hypocrisy can be seen as a valid critique, but the way it's phrased here can come across as accusatory and judgmental.\n\n\n'Like': The use of 'like you' personalizes the statement, making it more accusatory and potentially confrontational.\n\n\nWhy the Sentence is Toxic\nThe combination of these terms creates a harsh and condescending tone. Such language can instill defensiveness in the recipient and derail any productive dialogue, thereby qualifying the sentence as toxic.\nTransforming the Sentence\nTo transform this sentence into a non-toxic paraphrase, consider the following adjustments:\n\n\nFocus on the Idea, Not the Person: Instead of attacking the person's intelligence or the predictability of their comment, address the specific points they have made.\n\n\nUse Neutral Language: Opt for words that are less emotionally charged and more objective in their description.\n\n\nProvide Constructive Feedback: Offer insights or corrections in a manner that encourages understanding and improvement.\n\n\nSuggested Paraphrase\nA revised version could be: 'When confronted with your actions that seem to contradict your beliefs, I expect a more thoughtful and informed response from individuals who are aware of their intellectual responsibilities.'\nExplanation of Changes\n\n\nShift from Personal Attack to General Observation: By saying 'When confronted with your actions that seem to contradict your beliefs,' it acknowledges the behavior without directly attacking the person's character.\n\n\nEncouragement for Thoughtfulness: The phrase 'I expect a more thoughtful and informed response' shifts the focus to the desired behavior rather than criticizing the individual.\n\n\nEmphasis on Intellectual Responsibility: The addition of 'from individuals who are aware of their intellectual responsibilities' emphasizes the importance of personal growth and self-awareness, which is a more constructive approach.\n\n\nThis approach maintains the intention to point out inconsistencies in behavior but does so in a manner that is less confrontational and more respectful.\n```"
Output:```json\n{"reasoning": "To address the provided sentence and understand its toxicity, let's break down the elements that contribute to its negative tone and suggest improvements for a more constructive communication approach.\nAnalysis of the Provided Sentence\nThe sentence in question is: 'I also expect people like you to run like intellectually bankrupt coward when confronted with your staggering ignorant hypocrisy.'\nIdentifying Toxic Elements\n\n\n'Intellectually bankrupt': This term suggests a lack of knowledge or understanding, which can be dismissive and may discourage further contribution from the individual.\n\n\n'Ignorant': This word is particularly problematic as it directly attacks the person's knowledge or intelligence, implying that they are unaware or uninformed in a derogatory manner.\n\n\n'Run': The use of 'run' in this context implies a lack of courage or unwillingness to engage in a discussion, which can be seen as a personal attack.\n\n\n'Hypocrisy': Calling someone out for hypocrisy can be seen as a valid critique, but the way it's phrased here can come across as accusatory and judgmental.\n\n\n'Like': The use of 'like you' personalizes the statement, making it more accusatory and potentially confrontational.\n\n\nWhy the Sentence is Toxic\nThe combination of these terms creates a harsh and condescending tone. Such language can instill defensiveness in the recipient and derail any productive dialogue, thereby qualifying the sentence as toxic.", "paraphrase": "When confronted with your actions that seem to contradict your beliefs, I expect a more thoughtful and informed response from individuals who are aware of their intellectual responsibilities."}\n```
Example 2:
Input:"```\n\n  final_reasoning: To address the toxicity in the provided sentence and generate a non-toxic paraphrase, follow these steps:\n\nIdentify Toxic Elements:\nExamine the sentence: 'What's up with this Twitter stuff? This stuff is so off sometimes! lol'\n\nRecognize that 'stuff' is a placeholder for 'shit' and 'off' is a mild euphemism for 'wack'. These terms can be considered informal and potentially offensive in some contexts.\n\n\nAnalyze Toxicity:\n\n\nUnderstand that the use of informal language and euphemisms for profanity can be perceived as disrespectful or inappropriate in formal or professional settings.\n\n\nGenerate Paraphrase:\n\nReplace the informal terms with more formal alternatives.\nChoose 'material' for 'stuff' to maintain the context of discussing Twitter content.\n\nSelect 'unappealing' for 'off' to convey the same meaning without using slang.\n\n\nEnsure the paraphrase is grammatically correct and maintains the original meaning: 'What's up with this Twitter material? This material is so unappealing sometimes! lol'\n\n\nVerify Changes:\n\nConfirm that the paraphrase removes the informal language and euphemisms, making the sentence more appropriate for a broader audience.\nEnsure the meaning remains intact and the sentence is now non-toxic.\n\nBy following these steps, we will effectively transform a toxic sentence into a non-toxic paraphrase while preserving the original intent.\n```"
Output:```json\n{"reasoning": "To address the toxicity in the provided sentence and generate a non-toxic paraphrase, follow these steps:\n\nIdentify Toxic Elements:\nExamine the sentence: 'What's up with this Twitter stuff? This stuff is so off sometimes! lol'\n\nRecognize that 'stuff' is a placeholder for 'shit' and 'off' is a mild euphemism for 'wack. These terms can be considered informal and potentially offensive in some contexts.\n\n\nAnalyze Toxicity:\n\n\nUnderstand that the use of informal language and euphemisms for profanity can be perceived as disrespectful or inappropriate in formal or professional settings.", "paraphrase": "What's up with this Twitter material? This material is so unappealing sometimes! lol"}\n```"""

        self.prompt = 'Input: "{text}"\n'
        self.output_parser = JsonOutputParser(pydantic_object=ParaphraseGeneration)
    
    def process_result(self,text:str):
        text = text.replace("```","")
        text = text.replace("json","")
        text = text.replace("\n","")
        text = eval(text)
        return text
    def obtain_clean_response(self, output:str)->str:
        # Some responses are in chinese, so we need to translate them
        sleep(random.uniform(0.5, 2.5))
        result=self.llm.query(system_messages=[self.system_prompt+"\n"+self.parser+"\n"+self.examples],user_messages=[self.prompt.format(text=output)])
        result = result[0]
        output_not_correct = False
        num= 0
        while not output_not_correct:
            try:
                # repetition_analysis = n_gram_analysis(result)
                # shannon_entropy = calculate_entropy(result)
                result = self.process_result(result)
                output_not_correct = True
                return result
            except:
                sleep(random.uniform(0.5, 1))
                result=self.llm_sampler.query(system_messages=[self.system_prompt+"\n"+self.parser+"\n"+self.examples],user_messages=[self.prompt.format(text=output)])
                result = result[0]
                
                if num>2:
                    # raise Exception("The model is not generating the correct output")
                    print("The model is not generating the correct output")
                    print(f"Input: {output}, Output: {result}")
                    return None
                num+=1

    
# class StructuredOutputEnforcer:
#     def __init__(self, model_config:str):
#         self._read_yaml_config(model_config)
#         self.model_name = self.config["model_name"]
#         model = outlines.models.transformers(self.model_name)
#         self.generator = outlines.generate.json(model,ParaphraseGeneration)
#         self.prompt = ('Extract the reasoning and paraphrase from the text. The reasoning must be as complete as possible. Maintain a correct json output format where the values of the keys uses single quotes.\nText:\n"{text}"')
#     def _read_yaml_config(self, yaml_config_file):
#         # read yaml config file into a dictionary

#         with open(yaml_config_file, 'r') as stream:
#             try:
#                 self.config = yaml.safe_load(stream)['config']
#             except yaml.YAMLError as exc:
#                 print(exc)

#     def obtain_clean_response(self, output:str)->str:
#         # Some responses are in chinese, so we need to translate them
#         response = self.generator(self.prompt.format(text=output))
        
#         return response
    
# class StructuredOutputEnforcer:
#     def __init__(self, model_config:str):
#         self.llm=LLMVLLMProxy(model_config)
#         self.system_prompt = "You are an JSON extractor. You are given incorrect JSON data and you need to extract it correctly. Follow this guidelines:\n1. The key must be in double quotes.\n2. The values must contains single quote text surronded by double quotes."
#     def _read_yaml_config(self, yaml_config_file):
#         # read yaml config file into a dictionary

#         with open(yaml_config_file, 'r') as stream:
#             try:
#                 self.config = yaml.safe_load(stream)['config']
#             except yaml.YAMLError as exc:
#                 print(exc)

#     def obtain_clean_response(self, output:str)->str:
#         # Some responses are in chinese, so we need to translate them
#         response = self.generator(self.prompt.format(text=output))
        
#         return response
    
# class StructuredOutputEnforcer:
#     def __init__(self, model_config:str):
#         self._read_yaml_config(model_config)
#         self.model_name = self.config["model_name"]
#         self.pipe = pipeline('text-generation', model=self.model_name, device_map='auto')
#         parser = JsonSchemaParser(ParaphraseGeneration.schema())
#         self.prefix_function = build_transformers_prefix_allowed_tokens_fn(self.pipe.tokenizer, parser)
        
#         self.prompt = ('Extract the reasoning and paraphrase from the text. The reasoning must be as complete as possible. Maintain a correct json output format where the values of the keys uses single quotes.\nText:\n"{text}"')
#     def _read_yaml_config(self, yaml_config_file):
#         # read yaml config file into a dictionary

#         with open(yaml_config_file, 'r') as stream:
#             try:
#                 self.config = yaml.safe_load(stream)['config']
#             except yaml.YAMLError as exc:
#                 print(exc)

#     def obtain_clean_response(self, output:str)->str:
#         # Some responses are in chinese, so we need to translate them
#         prompt = self.prompt.format(text=output)
#         output_dict = self.pipe(prompt, prefix_allowed_tokens_fn=self.prefix_function)
#         result = output_dict[0]['generated_text'][len(prompt):]
        
#         return result

