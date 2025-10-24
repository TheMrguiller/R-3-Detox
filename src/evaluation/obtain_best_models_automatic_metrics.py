import pandas as pd
from scipy.optimize import linprog
import os
import sys
project_path=os.path.abspath(__file__).split('src')[0]
sys.path.append(project_path)
import argparse
pd.set_option('display.max_columns', None)


def get_top_rows(df, criteria, top_n=3, weights=None):
    """
    Get the top N rows from the DataFrame that satisfy the optimization criteria, with optional weighting.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        criteria (dict): A dictionary specifying the criteria for each column. 
                         Keys are column names, and values are either 'max' or 'min'.
        top_n (int): The number of top rows to retrieve.
        weights (dict): Optional. A dictionary specifying the weight for each column. 
                        Keys are column names, and values are the weights (default is 1 for all).

    Returns:
        pd.DataFrame: The top N rows that satisfy the criteria.
    """
    if weights is None:
        weights = {col: 1.0 for col in criteria.keys()}  # Default weight of 1.0 for all columns

    # Ensure all criteria columns exist in the DataFrame
    for col in criteria.keys():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' in criteria is not in the DataFrame.")

    # Convert criteria and weights into a weighted objective
    objective = []
    for col in df.columns:
        if col in criteria:
            weight = weights.get(col, 1.0)  # Default weight is 1.0 if not specified
            if criteria[col] == 'max':
                objective.append(-weight * df[col].values)  # Negate for maximization
            elif criteria[col] == 'min':
                objective.append(weight * df[col].values)  # Positive for minimization
            else:
                raise ValueError("Criteria values must be either 'max' or 'min'.")
        else:
            continue

    # Flatten the objective to match row selection
    c = sum(objective)

    # Constraint: At most `top_n` rows can be selected
    A_eq = [[1] * len(df)]  # Sum of selected rows must be 1
    b_eq = [1]
    bounds = [(0, 1) for _ in range(len(df))]  # Binary variables for row selection

    top_rows = []
    available_rows = df.copy()

    for _ in range(top_n):
        # Run the optimization
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            selected_row_index = result.x.argmax()
            top_rows.append(available_rows.iloc[selected_row_index])
            available_rows = available_rows.drop(index=available_rows.index[selected_row_index])

            # Update bounds and objective for remaining rows
            c = sum([(-weights.get(col, 1.0) * available_rows[col].values if criteria.get(col) == 'max'
                      else weights.get(col, 1.0) * available_rows[col].values)
                     for col in criteria.keys()])
            A_eq = [[1] * len(available_rows)]
            b_eq = [1]
            bounds = [(0, 1) for _ in range(len(available_rows))]
        else:
            break

    return pd.DataFrame(top_rows)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model",type=str,help="Folder we want to extract the paraphrase from",default="qwen2_5_7B")#llama3_1_8B,openO1,marco-o1,qwen2_5_7B
    model = args.parse_args().model
    for model in ["llama3_1_8B","openO1","marco-o1","qwen2_5_7B","capp","qwq_preview"]:
        folder_path = project_path + f"results/reports/paraphrase_automatic_metrics/"
        if os.path.exists(folder_path + f"{model}_APPDIA.csv"):
            df_APPDIA = pd.read_csv(folder_path + f"{model}_APPDIA.csv")
        else:
            df_APPDIA = None
        if os.path.exists(folder_path + f"{model}_paradetox.csv"):
            df_paradetox = pd.read_csv(folder_path + f"{model}_paradetox.csv")
        else:
            df_paradetox = None
        if os.path.exists(folder_path + f"{model}_parallel.csv"):
            df_parallel = pd.read_csv(folder_path + f"{model}_parallel.csv")
        else:
            df_parallel = None

        criteria = {
            'Free_bert_scores': 'max',
            'Free_bleu_scores': 'max',
            # 'Free_content_similarities': 'max',
            # 'Free_fluency_scores': 'max',
            # 'Free_style_transfer_scores': 'max',
            'Free_joint_score': 'max',
            'Ref_toxic_scores': 'min',
            # 'Ref_bert_scores': 'max',
            # 'Ref_rouge_scores': 'max',
            # 'Ref_bleu_scores': 'max',
        }
        weight = {
            'Free_bert_scores': 1,
            'Free_bleu_scores': 1,
            # 'Free_content_similarities': 3,
            # 'Free_fluency_scores': 1,
            # 'Free_style_transfer_scores': 1,
            'Free_joint_score': 2,
            'Ref_toxic_scores': 3,
            # 'Ref_bert_scores': 1,
            # 'Ref_rouge_scores': 1,
            # 'Ref_bleu_scores': 1,
        }
        

        names = ["APPDIA","paradetox","parallel"]
        # for df,name in zip([df_APPDIA,df_paradetox,df_parallel],names):
        #     if df is not None:
        #         if model == "capp":
        #             top_rows = get_top_rows(df, criteria, top_n=20,weights=weight)
        #             print(f"Top rows for dataframe {name}, model {model}:\n", top_rows)
        #         else:
        #             top_rows = get_top_rows(df, criteria, top_n=5,weights=weight)
        #             print(f"Top rows for dataframe {name}, model {model}:\n", top_rows)
        
        # Obtain median and std for each metric
        columns = ['Free_bert_scores', 'Free_bleu_scores', 'Free_content_similarities', 'Free_fluency_scores',
                   'Free_style_transfer_scores', 'Free_joint_score', 'Ref_toxic_scores', 'Ref_bert_scores',
                   'Ref_rouge_scores', 'Ref_bleu_scores']
        if df_APPDIA is not None:
            print(f"Median for APPDIA, model {model}:\n", df_APPDIA[columns].mean())
            print(f"Standard deviation for APPDIA, model {model}:\n", df_APPDIA[columns].std())
        if df_paradetox is not None:
            print(f"Median for paradetox, model {model}:\n", df_paradetox[columns].mean())
            print(f"Standard deviation for paradetox, model {model}:\n", df_paradetox[columns].std())
        if df_parallel is not None:
            print(f"Median for parallel, model {model}:\n", df_parallel[columns].mean())
            print(f"Standard deviation for parallel, model {model}:\n", df_parallel[columns].std())