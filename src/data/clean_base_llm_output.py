import pandas as pd
import re

if __name__ == "__main__":
    
    df = pd.read_csv("/global/home/TRI.LAN/guillermo.villate/Reasoner_Paraphraser/data/processed/final_paraphrases/basellm/openO1/openO1.csv")
    df["result"] = df["result"].apply(lambda text: re.search(r"<Output>(.*?)</Output>", text, re.DOTALL).group(1).strip() \
    if re.search(r"<Output>(.*?)</Output>", text, re.DOTALL) else None)

    df.to_csv("/global/home/TRI.LAN/guillermo.villate/Reasoner_Paraphraser/data/processed/final_paraphrases/basellm/openO1/openO1.csv", index=False)

    # The rest of the code would go here, but it is not provided in the snippet.