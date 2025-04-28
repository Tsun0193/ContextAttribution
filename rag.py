import os
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

load_dotenv()
nest_asyncio.apply()
warnings.filterwarnings("ignore")


document = 'documents/attention.pdf'
MODEL_NAME = "Llama-3.2-1B-Instruct"
model_name = "meta-llama/Llama-3.2-1B-Instruct" # 3.2 1B Instruct for faster inference, 3.1 8B for better performance


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


parser = LlamaParse(
    api_key=os.getenv("LLAMACLOUD_API_KEY"),
    num_workers=8,
    show_progress=True,
    result_type="markdown"
)


file = "documents/sample.pdf"
if not os.path.exists(file):
    raise FileNotFoundError(f"File {file} not found")


documents = parser.load_data(file)
nodes = splitter.get_nodes_from_documents(documents, show_progress=True)


storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)
vector_index = VectorStoreIndex(nodes=nodes, 
                         insert_batch_size=1024, 
                         storage_context=storage_context,
                         show_progress=True)                                                          


dense_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
sparse_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

res_synth = get_response_synthesizer()

class HybridRetriever(BaseRetriever):
    def __init__(self, dense_retriever: BaseRetriever = dense_retriever, 
                 sparse_retriever: BaseRetriever = sparse_retriever,
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


hybrid_retriever = HybridRetriever(dense_retriever=dense_retriever, sparse_retriever=sparse_retriever)
query_engine = RetrieverQueryEngine(retriever=hybrid_retriever, response_synthesizer=res_synth)


def rerank(query, nodes,
           model=model_name,
           top_k=None):
    context = "\n\n".join([node.node.get_content() for node in nodes])

    cc = ContextCiter.from_pretrained(
        model_name,
        context=context,
        query=query,
        device="cuda:3",
        num_ablations=32 if len(nodes) < 128 else 64
    )

    attributions = cc.get_attributions(as_dataframe=True, top_k=len(nodes) if top_k is None else top_k)

    if hasattr(attributions, "data"):
        attributions_df = attributions.data
    else:
        attributions_df = attributions

    segments = cc.sources
    score_list = attributions_df["Score"].tolist()

    node_scores = {}
    for node in nodes:
        node_text = node.node.get_content()
        cumulative_score = 0.0
        for seg, score in zip(segments, score_list):
            if seg.strip() and seg.strip() in node_text:
                cumulative_score += score
        node_scores[node.node.node_id] = cumulative_score

    reranked_nodes = sorted(nodes, key=lambda x: node_scores.get(x.node.node_id, 0.0), reverse=True)
    return reranked_nodes, cc.response, node_scores, attributions


def query_pdf(query: str) -> list:
    """ 
    Query the PDF document using the query engine."
    """
    global query_engine
    if query_engine is None:
        return "Please upload a PDF first."

    nodes = query_engine.retriever.retrieve(QueryBundle(query))
    reranked_nodes, cc_response, node_scores, attrs = rerank(query, nodes)

    top_nodes = reranked_nodes[:5]  # select the top 5 nodes
    final_context = "\n\n".join([node.node.get_content() for node in top_nodes])

    final_response = (
        f"{cc_response}\n\nTop Ranked Context:\n{final_context}\n\nNode Scores:\n{node_scores}"
    )

    if hasattr(attrs, "data"):
        attrs = attrs.data

    return final_response, attrs


res, attributions = query_pdf("What is the purpose of the printf function in C?")


attributions

