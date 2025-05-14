# app_streamlit.py

import os
import torch
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import nest_asyncio
from dotenv import load_dotenv
from scipy.stats import spearmanr

from llama_parse import LlamaParse
from llama_index.core import Document, VectorStoreIndex, get_response_synthesizer, StorageContext, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.node_parser import SemanticDoubleMergingSplitterNodeParser, LanguageConfig
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from context_cite import ContextCiter
from context_cite.utils import aggregate_logit_probs
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# 1. Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="ContextCite Document Analysis", layout="wide")
load_dotenv()
nest_asyncio.apply()
parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_TOKEN"))

# LlamaIndex splitter
config = LanguageConfig(language="english", spacy_model="en_core_web_md")
splitter = SemanticDoubleMergingSplitterNodeParser(
    initial_threshold=0.6,
    appending_threshold=0.5,
    merging_threshold=0.6,
    language_config=config,
    max_chunk_size=1024,
)

# Load LLM
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------------------------------------------------------
# 2. Helpers
# -----------------------------------------------------------------------------
def preprocess_document(uploaded_file) -> str:
    """Đọc PDF/TXT, trả về một chuỗi context"""
    if uploaded_file.type == "application/pdf":
        docs = parser.load_data(uploaded_file)
        text = " ".join([d.text for d in docs if len(d.text) >= 32])
    elif uploaded_file.type.startswith("text/"):
        text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type.")
        return ""
    return text

def plot_correlation(cc: ContextCiter) -> plt.Figure:
    preds = aggregate_logit_probs(cc._logit_probs).flatten()
    actual = cc._actual_logit_probs.flatten()
    corr, _ = spearmanr(preds, actual)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(preds, actual, alpha=0.3)
    lims = [min(preds.min(), actual.min()), max(preds.max(), actual.max())]
    ax.plot(lims, lims, '--', color='gray')
    ax.set_xlabel("Predicted log-prob")
    ax.set_ylabel("Actual log-prob")
    ax.set_title(f"Spearman ρ = {corr:.2f}")
    ax.grid(True)
    return fig

class HybridRetriever(BaseRetriever):
    def __init__(self, dense_retriever, sparse_retriever, mode="OR", **kwargs):
        super().__init__(**kwargs)
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.mode = mode

    def _retrieve(self, query_bundle):
        dense = self.dense_retriever.retrieve(query_bundle)
        sparse = self.sparse_retriever.retrieve(query_bundle)
        d_ids = {n.node.node_id for n in dense}
        s_ids = {n.node.node_id for n in sparse}
        all_nodes = {n.node.node_id: n for n in dense + sparse}
        if self.mode=="AND":
            ids = d_ids & s_ids
        else:
            ids = d_ids | s_ids
        return [all_nodes[i] for i in ids]

def analyze(context: str, query: str, top_k: int, n_ablate: int):
    # 1) chunk + index
    nodes = splitter.get_nodes_from_documents([Document(text=context)])
    storage = StorageContext.from_defaults()
    storage.docstore.add_documents(nodes)
    vector_index = VectorStoreIndex(nodes=nodes, storage_context=storage)

    # 2) retrievers
    dense_ret = VectorIndexRetriever(index=vector_index, similarity_top_k=top_k*2)
    sparse_ret = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k*2)
    hybrid = HybridRetriever(dense_retriever=dense_ret, sparse_retriever=sparse_ret, mode="OR")

    # 3) retrieve sync
    rnodes = hybrid.retrieve(QueryBundle(query))
    filtered_ctx = "\n\n".join([n.node.get_content() for n in rnodes])

    # 4) ContextCite
    cc = ContextCiter(
        model=model,
        tokenizer=tokenizer,
        context=filtered_ctx,
        query=query,
        num_ablations=n_ablate
    )
    df = cc.get_attributions(as_dataframe=True, top_k=top_k)
    fig = plot_correlation(cc)
    return df, fig

# -----------------------------------------------------------------------------
# 3. Streamlit UI
# -----------------------------------------------------------------------------
st.title("ContextCite Document Analysis")

uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf","txt"])
if uploaded:
    context = preprocess_document(uploaded)
    st.success("Document loaded. Length: {:,} chars".format(len(context)))
    with st.expander("Preview of context"):
        st.write(context[:2000] + "…")

    query = st.text_input("Enter your query:", value="What is Transformer?")
    top_k = st.slider("Top K chunks", 1, 10, 5)
    n_ablate = st.slider("Number of ablations", 16, 128, 64)
    if st.button("Analyze"):
        with st.spinner("Running ContextCite..."):
            df, fig = analyze(context, query, top_k, n_ablate)
        st.subheader("Attribution scores")
        st.dataframe(df)
        st.subheader("Predicted vs. Actual log-probability")
        st.pyplot(fig)