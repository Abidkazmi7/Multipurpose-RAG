from embeddings.models import crossencoder_model

model = crossencoder_model()

# Reranking function
def rerank(query, docs, top_k = 3):
    pairs = [[query, doc.page_content] for doc in docs]

    scores = model.predict(pairs)
    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key = lambda x : x[0], reverse = True)

    return [
        doc for score, doc in scored_docs[:top_k]
    ]

def rerank_docs(inputs):
    return rerank(
        query = inputs["question"],
        docs = inputs["docs"]
    )