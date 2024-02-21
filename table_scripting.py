from pathlib import Path
from collections import defaultdict 
import pandas as pd
import os
import re
import json

DATA_PATH = "./finetune_new_std"
DATASET = ["arxiv", "cora", "amazoncomputers", "amazonphoto", "coauthorcs", "flickr", "reddit2", "products"]
# DATASET = ["arxiv", "cora"]
#, 'GCN2_MLP', 'GAT_MLP', 'JKNET_MLP', 'GIN_MLP'
MODEL = ["GCN_MLP", "SAGE_MLP", "GAT_MLP", "GIN_MLP", "GCN", "SAGE", "GAT", "GIN"]
# MODEL = ["GCN_MLP", "SAGE_MLP", "GCN", "SAGE"]
# MODEL = ["MLP"]
# desired_keys = ["total_time"]
# desired_keys = ["bef_edit_tst_acc", "test_drawdown", ""]
# desired_keys = ["hop_drawdown"]
desired_keys = ["table_result", "highest_dd", "average_dd", "success_rate"]
# desired_keys = ["bef_edit_tst_acc", "test_drawdown", "success_rate", "average_dd", "highest_dd", "lowest_dd", "success_list", "table_result"]

# checkpoints = [str(x) for x in Path(saved_ckpt_path).glob(f"{cls.__name__}_*.pt")]
file_names = []
dir_path = os.path.dirname(os.path.realpath(__file__))

for dataset in DATASET:
    for model in MODEL:
        dataset_path = DATA_PATH + f'/{dataset}/GD'
        file_names.extend([str(x) for x in Path(dataset_path).glob(f"*_{model}_*.json")])

table_group = defaultdict(lambda: [])

for file_name in file_names:
    setting = None
    for model_name in MODEL:
        if model_name in file_name:
            setting = file_name.split(model_name)[0].split('/')[-1]
            break
    table_group[setting].append(file_name)

for k, v in table_group.items():
    result_files = v
    df = pd.DataFrame(index=MODEL, columns=DATASET)
    for file in result_files:
        # Extract model and dataset names from the file path
        model_name, dataset_name = None, None
        for model in MODEL:
            if model in file:
                model_name = model
                break
        for dataset in DATASET:
            if dataset in file:
                dataset_name = dataset
                break
        
        if model_name is not None and dataset_name is not None:
            with open(os.path.join(dir_path, file), 'r') as f:
                content = json.load(f)
            
            # Extract desired keys from the content
            content = content['batch_edit']
            values = {key: content.get(key, None) for key in desired_keys}
            df.at[model_name, dataset_name] = values
    df.to_csv(os.path.join(dir_path, f"{k}.csv"), index_label="Model/Dataset")
    print(df)