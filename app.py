print("Importing libraries...")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import openai
import chromadb
import warnings
import nest_asyncio
import pandas as pd

from llama_parse import LlamaParse
from llama_index.core import Document, VectorStoreIndex, get_response_synthesizer, StorageContext, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.node_parser import SemanticDoubleMergingSplitterNodeParser, LanguageConfig
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from typing import Literal, List, Optional
from dotenv import load_dotenv
from tqdm.auto import tqdm
from context_cite import ContextCiter
from context_cite.utils import aggregate_logit_probs
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Setting up environment...")

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.cuda.set_device(3)
openai.api_key = os.getenv("OPENAI_API_KEY")
    
config = LanguageConfig(language="english", spacy_model="en_core_web_md") # must download the model first
embed_model = OpenAIEmbedding()
splitter = SemanticDoubleMergingSplitterNodeParser(
    initial_threshold=0.6,
    appending_threshold=0.5,
    merging_threshold=0.6,
    language_config=config,
    max_chunk_size=1024,
)

print("Loading modules...")
load_dotenv()
warnings.filterwarnings("ignore")
parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_TOKEN"))

# Load LLM once
print("Initializing LLM model...")
model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("LLM model initialized!")

# Preprocess the uploaded file
def preprocess_document(file):
    if hasattr(file, 'name') and file.name.endswith(".pdf"):
        docs = parser.load_data(file.name)
        context = " ".join([doc.text for doc in docs if len(doc.text) >= 32])
    elif hasattr(file, 'name') and file.name.endswith(".txt"):
        with open(file.name, "r", encoding="utf-8") as f:
            context = f.read()
    else:
        raise ValueError("Unsupported file format. Please upload .pdf or .txt.")
    return context, "Document successfully preprocessed."

# Plotting function
def plot(cc: ContextCiter) -> plt.Figure:
    pred_logs = cc._logit_probs
    pred_logits = aggregate_logit_probs(pred_logs)
    actu_logits = cc._actual_logit_probs

    preds = pred_logits.flatten()
    actus = actu_logits.flatten()

    corr, _ = spearmanr(preds, actus)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(preds, actus, alpha=0.3, label="Context ablations")

    x_line = np.linspace(min(preds.min(), actus.min()), max(preds.max(), actus.max()), 100)
    plt.plot(x_line, x_line, '--', color='gray', label="y = x")

    plt.xlabel("Predicted log-probability")
    plt.ylabel("Actual log-probability")
    plt.title(f"Predicted vs. Actual log-probability\nSpearman correlation: {corr:.2f}")
    plt.legend()
    plt.grid(True)

    plt.close()
    return fig

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

# Analysis using preprocessed context
def analyze_document(context, query, top_k=5, num_ablations=64):
    nodes = splitter.get_nodes_from_documents([Document(text=context)])
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    vector_index = VectorStoreIndex(nodes=nodes, 
                                    insert_batch_size=1024, 
                                    storage_context=storage_context,
                                    show_progress=True)
    dense_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
    sparse_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

    hybrid_retriever = HybridRetriever(dense_retriever=dense_retriever, sparse_retriever=sparse_retriever)
    res_synth = get_response_synthesizer()
    query_engine = RetrieverQueryEngine(retriever=hybrid_retriever, response_synthesizer=res_synth)
    
    rnodes = query_engine.retriever.retrieve(QueryBundle(query))
    context = "\n\n".join([node.node.get_content() for node in rnodes])

    cc = ContextCiter(
        model=model,
        tokenizer=tokenizer,
        context=context,
        query=query,
        num_ablations=num_ablations
    )

    df = cc.get_attributions(as_dataframe=True, top_k=top_k)
    fig = plot(cc)
    return df, fig

# Gradio interface
print("Creating Gradio interface...")

with gr.Blocks() as demo:
    gr.Markdown("# ContextCite Document Analysis Demo")
    gr.Markdown("Upload a PDF or TXT document, preprocess it, then ask a question to analyze attribution.")

    file_input = gr.File(label="Upload Document (PDF or TXT)")
    context_state = gr.State()
    preprocess_status = gr.Textbox(label="Preprocessing Status", interactive=False)

    query_input = gr.Textbox(label="Query", value="What is Transformer?")
    top_k_slider = gr.Slider(1, 10, value=5, label="Top K Results")
    ablation_slider = gr.Slider(16, 128, value=64, label="Number of Ablations")

    output_df = gr.Dataframe(label="Attribution Results")
    output_plot = gr.Plot(label="Probability Correlation")

    file_input.change(
        fn=preprocess_document,
        inputs=[file_input],
        outputs=[context_state, preprocess_status]
    )

    analyze_btn = gr.Button("Analyze")
    analyze_btn.click(
        fn=analyze_document,
        inputs=[context_state, query_input, top_k_slider, ablation_slider],
        outputs=[output_df, output_plot]
    )

print("Gradio interface created successfully")
print("Launching demo...")

if __name__ == "__main__":
    demo.launch(share=True)
