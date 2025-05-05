from context_cite.context_citer import ContextCiter
from tqdm.auto import tqdm
from transformers.utils import logging as hf_logging
import pandas as pd
import json
import torch
import warnings
import os

# suppress warnings and HF progress bars
warnings.filterwarnings('ignore')
hf_logging.disable_progress_bar()

def compute_logprob(model, tokenizer, prompt: str, continuation: str, device: str) -> float:
    """
    Compute the total log-probability of `continuation` given `prompt` under the model.
    Returns the sum of token-level log-probs.
    """
    full_text = prompt + continuation
    encoded = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask

    # Compute prompt length accurately
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    prompt_len = len(prompt_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Compute log-probs
    log_probs = torch.nn.functional.log_softmax(logits[:, :-1], dim=-1)
    labels = input_ids[:, 1:]

    continuation_labels = labels[:, prompt_len-1:]  # offset by 1 due to shifted labels
    continuation_log_probs = log_probs[:, prompt_len-1:].gather(-1, continuation_labels.unsqueeze(-1)).squeeze(-1)

    return float(continuation_log_probs.sum())


# ---- load TydiQA development set ----
file_path = "/data_hdd_16t/duydang/ContextAttribution/data/tydiqa-v1.0-dev.jsonl"
english_samples = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading JSONL"):
        sample = json.loads(line)
        if sample.get('language', '').lower() == 'english':
            english_samples.append(sample)
print(f"Collected {len(english_samples)} English samples.")

# sort by context length and take a subset
data = sorted(english_samples, key=lambda x: len(x['document_plaintext']))[128:256]

# ---- initialize ContextCiter ----
model_name    = "meta-llama/Llama-3.2-1B-Instruct"
# model_name    = "meta-llama/Llama-3.2-3B-Instruct"
# model_name    = "meta-llama/Llama-3.1-8B-Instruct"
num_ablations = 128
cc = ContextCiter.from_pretrained(
    model_name,
    context="",      # placeholder, will be set per-sample
    query="",        # placeholder, will be set per-sample
    device="cuda", # change here if needed
    solver="lasso",
    num_ablations=num_ablations
    # quantized=True
)

results_dir = "results/topk_drop/tydiqa"
os.makedirs(results_dir, exist_ok=True)
csv_path = f"{results_dir}/{model_name.replace('/', '-')}_{num_ablations}abls_{len(data)}samples.csv"

write_header = not os.path.exists(csv_path)

for idx, sample in enumerate(tqdm(data, desc="Evaluating samples")):
    row = {"id": sample.get('example_id', '')}
    try:
        query        = sample['question_text']
        full_context = sample['document_plaintext']

        cc.reset()
        cc.context = full_context
        cc.query   = query

        prompt_full = cc.prompt_template.format(context=full_context, query=query)
        enc_full = cc.tokenizer(prompt_full, return_tensors="pt", padding=True, truncation=True).to(cc.model.device)
        out_ids  = cc.model.generate(**enc_full, pad_token_id=cc.tokenizer.eos_token_id)[0]
        n_prompt = enc_full.input_ids.shape[1]
        resp_ids = out_ids[n_prompt:]
        continuation = cc.tokenizer.decode(resp_ids, skip_special_tokens=True)

        lp_full = compute_logprob(cc.model, cc.tokenizer, prompt_full, continuation, cc.model.device)

        for k in [1, 3, 5]:
            at_df = cc.get_attributions(as_dataframe=True, verbose=False).data
            topk  = at_df.nlargest(k, 'Score')['Source'].tolist()

            ablated = full_context
            for src in topk:
                ablated = ablated.replace(src, "")
            prompt_abl = cc.prompt_template.format(context=ablated, query=query)
            lp_abl     = compute_logprob(cc.model, cc.tokenizer, prompt_abl, continuation, cc.model.device)
            row[f"k={k}"] = abs(lp_full - lp_abl)

    except Exception as e:
        # Log or print the error if needed
        print(f"[Sample {idx}] Error: {e}")
        # Fallback values
        for k in [1, 3, 5]:
            row[f"k={k}"] = 0.0

    # Write to CSV immediately
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", header=write_header, index=False)
    write_header = False