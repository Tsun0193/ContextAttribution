from context_cite.context_citer import ContextCiter
from tqdm.auto import tqdm
from transformers.utils import logging as hf_logging
from huggingface_hub import login

import pandas as pd
import warnings
import json
import re
import sys
import torch


warnings.filterwarnings('ignore')
hf_logging.disable_progress_bar()

file_path = "/data_hdd_16t/duydang/ContextAttribution/data/tydiqa-v1.0-dev.jsonl"

data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        sample = json.loads(line)
        if sample['language'].lower() == 'english':
            data.append(sample)

print(f"Collected {len(data)} english samples.")

model_name = "meta-llama/Llama-3.1-8B-Instruct"
n_ablations = 32


res = pd.DataFrame(columns=["k=1", "k=3", "k=5"])
cc = ContextCiter.from_pretrained(
    model_name,
    device="cuda:3",
    solver="lasso",
    num_ablations=n_ablations,
    quantized=True
)

for sample in tqdm(data[:16]):
    temp = pd.Series(index=["k=1", "k=3", "k=5"], dtype=float)
    query = sample['question_text']
    full_context = sample['document_plaintext']

    for k in [1, 3, 5]:
        cc.context = full_context
        cc.query = query
        attributions = cc.get_attributions(as_dataframe=True, verbose=False).data
        top_k_sources = attributions.nlargest(k, 'Score')["Source"]
        top_score = attributions.iloc[0].Score / attributions.Score.sum()

        ablated_context = full_context
        for src in top_k_sources:
            ablated_context = ablated_context.replace(src, "")
        
        cc.reset()

        cc.context = ablated_context
        cc.query = query
        _attributions = cc.get_attributions(as_dataframe=True, verbose=False).data
        _top_score = _attributions.iloc[0].Score / _attributions.Score.sum()

        temp[f'k={k}'] = top_score - _top_score

        cc.reset()

    res = pd.concat([res, temp.to_frame().T], ignore_index=True)

res.to_csv(f'results/Llama-3.1-8B_{n_ablations}ablations.csv')