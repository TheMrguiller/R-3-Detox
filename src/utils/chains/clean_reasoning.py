import sys
import os
proyect_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(proyect_path)
from src.utils.llms.llm_vllm_server_proxy import LLMVLLMProxy
from src.utils.llms.llm_vllm_offline_proxy import LLMVLLMOfflineProxy
from src.utils.structured_output.reasoningcleaning import ReasoningCleaning
import yaml
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from tqdm import tqdm
import logging
from transformers import AutoTokenizer
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class LLMReasoningCleaning:
    def __init__(self, yaml_config_file,offline=True) -> None:
        if offline:
            self.llm = LLMVLLMOfflineProxy(yaml_config_file)
        else:
            self.llm = LLMVLLMProxy(yaml_config_file)
        # self.llm = LLMVLLMOfflineProxy(yaml_config_file)
        with open(proyect_path + "src/utils/llms/prompts/extract_reasoning_prompt_toxic.yaml", "r") as file:
            toxic_prompt_config = yaml.safe_load(file) 
        with open(proyect_path + "src/utils/llms/prompts/extract_reasoning_prompt_non_toxic.yaml", "r") as file:
            non_toxic_prompt_config = yaml.safe_load(file)
        self.toxic_system_messages = toxic_prompt_config["system_message"].replace("\\n","\n").replace("\\t","\t").replace("\n ","\n")
        self.non_toxic_system_messages = non_toxic_prompt_config["system_message"].replace("\\n","\n").replace("\\t","\t").replace("\n ","\n")
        self.parser = JsonOutputParser(pydantic_object=ReasoningCleaning)
        self.user_messages = toxic_prompt_config["prompt"].replace("\\n","\n").replace("\n ","\n")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B")
        
    def query(self,labels:List[float],reasonings:List[str],batch_size:int=1)->List[str]:
        system_messages = []
        for label in labels:
            if label == 1.0:
                system_messages.append(self.toxic_system_messages+f"\n{self.parser.get_format_instructions()}")
            else:
                system_messages.append(self.non_toxic_system_messages+f"\n{self.parser.get_format_instructions()}")
        user_messages= []
        for i,reasoning in enumerate(reasonings):
            reasoning = reasoning.replace('"',"'")
            token_length=len(self.tokenizer.tokenize(reasoning))
            if token_length > int(self.llm.config["max_tokens"]/2): # We may have cases of alucination
                logger.info(f"Reasoning too long to handle: {token_length},Response number: {i}")
                reasoning = " ".join(reasoning.split()[:1024])
            user_messages.append(self.user_messages.format(reasoning=reasoning))
        final_response=[]
        for idx in tqdm(range(0, len(user_messages), batch_size)):
            system_messages_batch = system_messages[idx:idx+batch_size]
            user_messages_batch = user_messages[idx:idx+batch_size]
            labels_batch = labels[idx:idx+batch_size]
            final_response_batch = [""]*len(user_messages_batch)
            user_messages_batch_copy = []
            system_messages_batch_copy = []
            for i in range(0,len(user_messages_batch)):
                if "```python" in user_messages_batch[i]:
                    final_response_batch[i]=reasonings[idx+i]
                else:
                    final_response_batch[i]=""
                    user_messages_batch_copy.append(user_messages_batch[i])
                    system_messages_batch_copy.append(system_messages_batch[i])
            if len(user_messages_batch_copy) > 0:
                response = self.llm.query(system_messages_batch_copy, user_messages_batch_copy)
                for i, resp in enumerate(final_response_batch):
                    if resp == "":
                        resp = response[0]
                        response.pop(0)
                        try:
                            #reason = self.parser.parse(resp)["reasoning"]
                            if "```json" in resp and "```" in resp:
                                reason = self.extract_json_block_with_tags(resp)
                            else:
                                reason = self.extract_content_inside_braces(resp)
                            reason = reason.replace("\n","").replace("```json","").replace("```","")
                            reason = eval(reason)["reasoning"]
                            final_response_batch[i] = reason
                        except Exception as e:
                            logger.info(f"Error parsing the response: {e}")
                            print(f"Error parsing the response: {resp}")
                            print(f"Response: {user_messages_batch[i]}")
                            final_response_batch[i] = None
            final_response.extend(final_response_batch)
        return final_response
    
    def extract_json_block_with_tags(self,text):
        # Regular expression to capture text including ```json and ```
        match = re.search(r'```json.*?```', text, re.DOTALL)
        if match:
            return match.group(0).strip()  # Return the entire matched block including the tags
        return None

    def extract_content_inside_braces(self,text):
        # Regular expression to capture the content inside the largest pair of curly braces
        match = re.search(r'\{(.*)\}', text, re.DOTALL)
        if match:
            return '{' + match.group(1).strip() + '}'  # Return the matched content wrapped in {}
        return None
    

if __name__ == "__main__":
    
    llm_reasoning_cleaning = LLMReasoningCleaning(proyect_path + "src/utils/llms/configs/qwen2_5.yaml")
    labels = [1,0]
    reasoning = [
       """Alright, so I've got this sentence: "Jesus fucking Christ these girls are going places literally and figuratively." And I need to figure out why it's considered toxic and how to make a non-toxic version of it. The possible toxic words highlighted are 'Jesus' and 'fucking', but I need to think about this carefully.

First off, let's look at the words. "Jesus fucking Christ" is a profane expression. It's using religious figures' names in a blasphemous way, which can be offensive to people who hold those figures sacred. So, that's one aspect. Then there's the term "these girls are going places," which seems somewhat positive, but the addition of "literally and figuratively" might be sarcasm or could have a double meaning.

I need to consider the context in which this is said. If it's meant to be complimentary, like saying the girls are successful and achieving their goals, then maybe the issue is primarily with the profanity. But if it's meant derogatorily, perhaps implying that they're going places in a negative sense, like getting into trouble, then there's an additional layer of toxicity.

Given that "Jesus fucking Christ" is a strong expletive, it's likely that its use is what makes the sentence toxic. People who are sensitive to profanity or who come from religious backgrounds might find this offensive. Also, the combination of religious profanity and a statement about girls going places could potentially be seen as sexist, depending on the tone and context.

Now, to make a non-toxic version of this sentence, I need to remove or soften the offensive elements. The non-toxic paraphrase provided is "Jesus Christ these girls are going places literally and figuratively." So, the change here is removing the word "fucking" from the expression.

By doing this, the profanity is reduced, making it less offensive. "Jesus Christ" is still an expletive, but it's milder compared to "Jesus fucking Christ." This change likely makes the sentence less toxic because it's less provocative and less likely to offend people who are sensitive to strong language.

Additionally, if the original sentence had a negative connotation, removing the intensifier "fucking" might also tone down the negativity. However, without more context, it's hard to say for sure. But generally, reducing profanity is a step towards making the language more acceptable and less toxic.

So, in summary, the provided sentence is toxic mainly due to the use of strong profanity, specifically the phrase "Jesus fucking Christ." By removing "fucking," the paraphrase softens the language, making it less offensive and therefore less toxic.""",
        """I've been asked to analyze this sentence to determine why it's not considered toxic, even though there are some words that might be seen as negative. The sentence is: "You right, lame. They are burying or hiding him. Lots of places you can hid messed up POs." And the possible toxic word noted is 'lame.'

First, I need to understand what makes language toxic. Toxic language usually includes harmful, offensive, or inappropriate words or phrases that can degrade, insult, or harm someone's reputation or feelings. It's important to consider the context in which words are used because the same word can have different implications in different situations.

Looking at the sentence, there are a few elements that might raise flags:

1. The word "lame" is used, which can be considered offensive or derogatory, especially when applied to a person or their actions.

2. Phrases like "burying or hiding him" and "messed up POs" could imply some sort of wrongdoing or cover-up, which might be seen as accusatory or conspiratorial.

However, without more context, it's challenging to definitively label this sentence as toxic. Here's why:

- The use of "lame" could be interpreted as dismissive or critical, but it's not directly targeted at a person; it's more likely referring to an idea or situation.

- The phrases about burying or hiding someone and hiding problematic purchase orders (POs) seem to be part of a discussion or speculation about something, possibly in a workplace or organizational setting.

- The tone isn't explicitly hostile or abusive; it's more suggestive of frustration or disagreement with how things are being handled.

Given these points, while the sentence contains language that could be perceived as negative or critical, it doesn't cross the line into toxic language because it doesn't directly attack someone's character or use overtly harmful language. It's more about expressing dissatisfaction with a situation.

To further support this, let's consider that in many professional or casual conversations, people use informal or critical language without intending to be toxic. The key is whether the language is used to degrade or harm versus to express disagreement or frustration.

In conclusion, although the sentence includes potentially negative language like "lame," it doesn't qualify as toxic because it doesn't directly insult or harm someone in a malicious way. The context appears to be more about expressing discontent with a situation rather than attacking an individual."""
    ]
    print(llm_reasoning_cleaning.query(labels,reasoning))


