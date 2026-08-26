from langchain_community.retrievers import BM25Retriever

def bm25_retriever(documents):
    # Create retriever object
    retriever = BM25Retriever.from_documents(
        documents,
        k = 4
    )

    return retriever