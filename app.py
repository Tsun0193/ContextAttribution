print("Importing libraries...")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import os

from llama_cloud_services import LlamaParse
from dotenv import load_dotenv
from context_cite import ContextCiter
from context_cite.utils import aggregate_logit_probs
from scipy.stats import spearmanr

print("Importing complete")
print("Loading modules")
load_dotenv()
parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_TOKEN"))

def plot(cc: ContextCiter) -> plt.Figure:
    pred_logs = cc._logit_probs
    pred_logits = aggregate_logit_probs(pred_logs)
    actu_logits = cc._actual_logit_probs

    preds = pred_logits.flatten()
    actus = actu_logits.flatten()
    assert len(preds) == len(actus), f"{len(preds)} != {len(actus)}"

    # Compute Spearman correlation
    corr, _ = spearmanr(preds, actus)

    # Create figure explicitly
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(preds, actus, alpha=0.3, label="Context ablations")
    
    # Plot reference line
    x_line = np.linspace(min(preds.min(), actus.min()), 
                        max(preds.max(), actus.max()), 100)
    plt.plot(x_line, x_line, '--', color='gray', label="y = x")
    
    # Add labels and styling
    plt.xlabel("Predicted log-probability")
    plt.ylabel("Actual log-probability")
    plt.title(f"Predicted vs. Actual log-probability\nSpearman correlation: {corr:.2f}")
    plt.legend()
    plt.grid(True)
    
    # Close plot to prevent memory leaks and return figure
    plt.close()
    return fig

# Define core function
def analyze_document(file, query, top_k=5, num_ablations=64):
    print("analyze_document() called")
    print(f"Received file: {file.name}")
    print(f"Query: {query}, Top K: {top_k}, Num Ablations: {num_ablations}")

    # Process input file
    if file.name.endswith(".pdf"):
        try:
            docs = parser.load_data(file.name)
            context = " ".join([doc.text for doc in docs if len(doc.text) >= 32])
        except Exception as e:
            print("Error reading PDF file:", e)
            raise
    elif file.name.endswith(".txt"):
        try:
            with open(file.name, "r") as f:
                context = f.read()
        except Exception as e:
            print("Error reading TXT file:", e)
            raise
    else:
        raise ValueError(f"Unsupported file format: {file.name}")

    print("Successfully read the file and constructed context.")

    try:
        cc = ContextCiter.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            context=context,
            query=query,
            device="cuda",
            num_ablations=num_ablations
        )
    except Exception as e:
        print("Error initializing ContextCiter:", e)
        raise

    print("ContextCiter created successfully.")

    try:
        df = cc.get_attributions(as_dataframe=True, top_k=top_k)
    except Exception as e:
        print("Error getting attributions:", e)
        raise

    print("Attributions obtained, building plot...")

    try:
        fig = plot(cc)
    except Exception as e:
        print("Error generating plot:", e)
        raise

    print("Returning DataFrame and plot.")
    return df, fig


print("Modules loaded successfully")
print("Creating Gradio interface...")

demo = gr.Interface(
    fn=analyze_document,
    inputs=[
        gr.File(label="Upload Document (PDF or TXT)"),
        gr.Textbox(label="Query", value="What is Transformer?"),
        gr.Slider(1, 10, value=5, label="Top K Results"),
        gr.Slider(16, 128, value=64, label="Number of Ablations")
    ],
    outputs=[
        gr.Dataframe(label="Attribution Results"),
        gr.Plot(label="Probability Correlation")
    ],
    title="ContextCite Document Analysis Demo",
    description="Upload a document and ask questions about its content"
)
print("Gradio interface created successfully")
print("Launching demo...")

if __name__ == "__main__":
    demo.launch(share=True)