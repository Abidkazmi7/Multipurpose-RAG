from ingestion.pdf_loader import load_pdf
from ingestion.youtube_loader import load_youtube
from processing.pdf_chunker import text_chunker
from processing.youtube_chunker import yt_semantic_chunk
from vectorstore.chroma_db import vector_database
from embeddings.models import huggingface_model
from retrieval.multi_query import create_multi_query, get_unique_union
from models.groq import query_llm

def build_retriever(chunks):
    model = huggingface_model()

    # Create embeddings for all chunks
    vector_db = vector_database(chunks, model)

    # Obtain top k relevant chunks
    retrieved_chunks = vector_db.as_retriever()

    return retrieved_chunks

# Document retrieval
def doc_retriever(pdf_path):
    docs = load_pdf(pdf_path)
    chunks = text_chunker(docs)
    retrieved_chunks = build_retriever(chunks)

    return retrieved_chunks

# Youtube video retriever
def youtube_retriever(url):
    data = load_youtube(url)
    chunks = yt_semantic_chunk(data["transcript"])
    retrieved_chunks = build_retriever(chunks)\
    
    return retrieved_chunks

def retrieval_chain(retriever):
    # Retrieving documents for each query
    query_generation_chain = create_multi_query(query_llm)

    retrieval_chain = query_generation_chain | retriever.map() | get_unique_union

    return retrieval_chain