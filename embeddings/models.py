from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import os

os.environ["HF_TOKEN"] = "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"

# Loads and returns the embedding model used to convert chunks into vector embeddings
def huggingface_model():
    model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

    return model

# Cross encoder model for reranking
def crossencoder_model(): 
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return model