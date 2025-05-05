from context_cite.context_citer import ContextCiter
from tqdm.auto import tqdm
from dotenv import load_dotenv
from typing import List, Literal, Optional
from transformers.utils import logging as hf_logging
from llama_index.core import Document, VectorStoreIndex, get_response_synthesizer, StorageContext, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.node_parser import SemanticDoubleMergingSplitterNodeParser, LanguageConfig
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from embedder import LocalEmbedding
from scipy.stats import spearmanr
import pandas as pd
import json
import torch
import os, warnings
import gc
import time
import random

# suppress warnings and HF bars
warnings.filterwarnings('ignore')
hf_logging.disable_progress_bar()
embed_model = LocalEmbedding()
load_dotenv()
random.seed(42)
torch.manual_seed(42)

# === RAG CONFIG ===
config = LanguageConfig(language="english", spacy_model="en_core_web_md")
splitter = SemanticDoubleMergingSplitterNodeParser(
    initial_threshold=0.75,
    appending_threshold=0.65,
    merging_threshold=0.75,
    language_config=config,
    max_chunk_size=256,
)

res_synth = get_response_synthesizer()

# ---- helper for computing exact log‐probabilities ----
def compute_logprob(model, tokenizer, prompt: str, continuation: str, device: str) -> float:
    full = prompt + continuation
    enc = tokenizer(full, return_tensors="pt", truncation=True).to(device)
    # find prompt length in tokens
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    plen = len(prompt_ids)
    with torch.no_grad():
        out = model(**enc).logits
    logp = torch.nn.functional.log_softmax(out[:, :-1], dim=-1)
    labels = enc.input_ids[:, 1:]
    cont_labels = labels[:, plen-1:]
    lp = logp[:, plen-1:].gather(-1, cont_labels.unsqueeze(-1)).squeeze(-1)
    return float(lp.sum())

# ---- load TydiQA dev ----
file_path = "data/hotpotqa_dev.jsonl"
samples = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading JSONL"):
        sample = json.loads(line)
        samples.append(sample)
print(f"Collected {len(samples)} samples.")

# take a middle slice by context length
random.shuffle(samples)
data = sorted(samples, key=lambda x: len(x["context"]["sentences"]))[64:128]
print(f"Using {len(data)} samples.")

# ---- init ContextCiter once ----
model_name    = "meta-llama/Llama-3.2-1B-Instruct"
num_ablations = 256
cc = ContextCiter.from_pretrained(
    model_name,
    context="",        # overwritten per‐sample
    query="",
    device="cuda:2",  
    solver="lasso",
    num_ablations=num_ablations
)

class HybridRetriever(BaseRetriever):
    def __init__(self, dense_retriever: BaseRetriever, 
                 sparse_retriever: BaseRetriever,
                 mode: Literal["AND", "OR"] = "OR",
                 **kwargs) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.mode = mode

        super().__init__(**kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        dense_res = self.dense_retriever.retrieve(query_bundle)
        sparse_res = self.sparse_retriever.retrieve(query_bundle)

        dense_ids = {n.node.node_id for n in dense_res}
        sparse_ids = {n.node.node_id for n in sparse_res}

        combined_ids = {n.node.node_id: n for n in dense_res}
        combined_ids.update({n.node.node_id: n for n in sparse_res})

        if self.mode == "AND":
            ids = dense_ids.intersection(sparse_ids)

        elif self.mode == "OR":
            ids = dense_ids.union(sparse_ids)

        else:
            raise ValueError("Invalid mode. Must be either 'AND' or 'OR'.")
        
        retrieved_nodes = [combined_ids[id] for id in ids]
        return retrieved_nodes

# output CSV
out_dir = "results/lds/hotpotqa"
os.makedirs(out_dir, exist_ok=True)
csv_path = f"{out_dir}/hybrid_{model_name.replace('/','-')}_{num_ablations}abls_{len(data)}samples.csv"
write_header = not os.path.exists(csv_path)

# === MAIN LOOP ===
start_time = time.time()
for idx, sample in enumerate(tqdm(data, desc="Eval")):
    row = {"id": sample.get("example_id","")}
    try:
        q = sample["question"]
        titles = sample["context"]["title"]
        docs = sample["context"]["sentences"]
        full_ctx_parts = []
        for title, sentences in zip(titles, docs):
            content = " ".join(sentences).strip()
            full_ctx_parts.append(f"Title: {title}\nContent: {content}")
        full_ctx = "\n\n".join(full_ctx_parts)

        doc = Document(text=full_ctx)
        storage = StorageContext.from_defaults()
        nodes = splitter.get_nodes_from_documents([doc])
        storage.docstore.add_documents(nodes)
        
        vs_idx = VectorStoreIndex(
            nodes,
            insert_batch_size=1024,
            storage_context=storage,
            embed_model=embed_model
        )
        dense_ret  = VectorIndexRetriever(vs_idx, similarity_top_k=5)
        sparse_ret = BM25Retriever(nodes=nodes, similarity_top_k=5)
        hybrid_ret = HybridRetriever(dense_ret, sparse_ret, mode="OR")
        qe = RetrieverQueryEngine(retriever=hybrid_ret, response_synthesizer=res_synth)
        rnodes = qe.retriever.retrieve(QueryBundle(q))
        rctx = "\n\n".join([n.node.get_content() for n in rnodes])

        cc.reset()
        cc.context = rctx
        cc.query   = q

        attrs = cc.get_attributions(as_dataframe=False, verbose=False)
        pred_logits = cc._pred_logit_probs
        actu_logits = cc._actual_logit_probs

        preds = pred_logits.flatten()
        actus = actu_logits.flatten()
        assert len(preds) == len(actus), f"{len(preds)} != {len(actus)}"

        corr, _ = spearmanr(preds, actus)
        row["corr"] = corr

        del vs_idx
        del hybrid_ret, sparse_ret, dense_ret
        del qe, rctx, rnodes
        del doc, nodes
        del storage
        gc.collect()
        torch.cuda.empty_cache()

        # append to CSV
        df = pd.DataFrame([row])
        df.to_csv(csv_path, mode="a", header=write_header, index=False)
        write_header = False
    
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        row["corr"] = 0
        df = pd.DataFrame([row])
        df.to_csv(csv_path, mode="a", header=write_header, index=False)
        write_header = False

print("Total time:", time.time() - start_time)