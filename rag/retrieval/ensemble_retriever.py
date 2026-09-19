from langchain_classic.retrievers import EnsembleRetriever

def ensemble_retriever(bm25_retriever, vector_retriever):
    retriever = EnsembleRetriever(
        retrievers = [bm25_retriever, vector_retriever],
        weights = [0.5, 0.5]
    )

    return retriever