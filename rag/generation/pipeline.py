from rag.ingestion.pdf_loader import load_pdf
from rag.ingestion.youtube_loader import load_youtube
from rag.ingestion.web_loader import load_webpage

from rag.chunking.pdf_chunker import text_chunker
from rag.chunking.youtube_chunker import yt_semantic_chunk
from rag.chunking.web_chunker import webpage_chunker

from rag.vectorstore.chroma_db import chroma_retriever
from rag.vectorstore.parent_store import build_parent_store

from rag.retrieval.multi_query import create_multi_query, get_unique_union
from rag.retrieval.bm25 import bm25_retriever
from rag.retrieval.ensemble_retriever import ensemble_retriever

from rag.models.language_models import query_llm

def build_retriever(chunks, embedding_model):
    # Create semantic retriever
    semantic_retriever = chroma_retriever(chunks, embedding_model)

    # Create keyword retriever
    keyword_retriever = bm25_retriever(chunks)

    # Build retriever object
    retriever = ensemble_retriever(
        keyword_retriever,
        semantic_retriever
    )

    return retriever

# Document retrieval object
def doc_retriever(pdf_path, embedding_model):
    # Contains section-level splits
    docs = load_pdf(pdf_path)

    # Parent-child chunking
    parent_chunks, child_chunks = text_chunker(docs)

    # Store parents for later lookup
    parent_store = build_parent_store(parent_chunks)

    # Retrievers search child chunks
    retriever = build_retriever(child_chunks, embedding_model)

    return retriever, parent_store

# Youtube video retriever object
def youtube_retriever(url, embedding_model):
    data = load_youtube(url)
    chunks = yt_semantic_chunk(data["transcript"], embedding_model)
    retriever = build_retriever(chunks, embedding_model)
    
    return retriever

# Webpage retriever object
def webpage_retriever(url, embedding_model):
    docs = load_webpage(url)
    chunks = webpage_chunker(docs)
    retriever = build_retriever(chunks, embedding_model)
    
    return retriever

def retrieval_chain(retriever):
    # Retrieving documents for each query
    query_generation_chain = create_multi_query(query_llm)

    retrieval_chain = query_generation_chain | retriever.map() | get_unique_union

    return retrieval_chain