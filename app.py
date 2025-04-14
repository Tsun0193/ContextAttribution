print("Importing libraries...")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from llama_cloud_services import LlamaParse
from dotenv import load_dotenv
from context_cite import ContextCiter
from context_cite.utils import aggregate_logit_probs
from scipy.stats import spearmanr

# Import transformers here so we can load the model once
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Importing complete")
print("Loading modules...")
load_dotenv()
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

# Analysis using preprocessed context
def analyze_document(context, query, top_k=5, num_ablations=64):
    cc = ContextCiter(
        model=model,
        tokenizer=tokenizer,
        context=context,
        query=query,
        num_ablations=num_ablations,
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
