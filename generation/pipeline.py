from ingestion.pdf_loader import load_pdf
from ingestion.youtube_loader import load_youtube
from processing.pdf_chunker import text_chunker
from processing.youtube_chunker import yt_semantic_chunk
from vectorstore.chroma_db import chroma_retriever
from vectorstore.parent_store import build_parent_store
from embeddings.models import huggingface_model
from retrieval.multi_query import create_multi_query, get_unique_union
from retrieval.bm25 import bm25_retriever
from retrieval.ensemble_retriever import ensemble_retriever
from models.groq import query_llm

def build_retriever(chunks):
    model = huggingface_model()

    # Create semantic retriever
    semantic_retriever = chroma_retriever(chunks, model)

    # Create BM25 retriever
    keyword_retriever = bm25_retriever(chunks)

    # Build retriever object
    retriever = ensemble_retriever(
        keyword_retriever,
        semantic_retriever
    )

    return retriever

# Document retrieval object
def doc_retriever(pdf_path):
    docs = load_pdf(pdf_path)

    # Parent-child chunking
    parent_chunks, child_chunks = text_chunker(docs)

    # Store parents for later lookup
    parent_store = build_parent_store(parent_chunks)

    # Retrievers search child chunks
    retriever = build_retriever(child_chunks)

    return retriever, parent_store

# Youtube video retriever object
def youtube_retriever(url):
    data = load_youtube(url)
    chunks = yt_semantic_chunk(data["transcript"])
    retriever = build_retriever(chunks)
    
    return retriever

def retrieval_chain(retriever):
    # Retrieving documents for each query
    query_generation_chain = create_multi_query(query_llm)

    retrieval_chain = query_generation_chain | retriever.map() | get_unique_union

    return retrieval_chain