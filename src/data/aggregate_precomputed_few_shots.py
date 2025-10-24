import json
import glob
import sys
import os
project_path = os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)

if __name__=="__main__":
    files = glob.glob(project_path+"data/processed/precomputed_few_shots_examples/*.json")
    if project_path+"data/processed/precomputed_few_shots_examples/precomputed_few_shot_examples.json" in files:
        files.remove(project_path+"data/processed/precomputed_few_shots_examples/precomputed_few_shot_examples.json")
    files.sort()
    dict_final={}
    for file in files:
        with open(file) as f:
            data = json.load(f)
        dict_final.update(data)
    keys = list(dict_final.keys())
    with open(project_path+"data/processed/precomputed_few_shots_examples/precomputed_few_shot_examples.json","w") as f:
        json.dump(dict_final,f)