from backend.resources import resources

print("Backend started.")
print("Embedding model:", type(resources.embedding_model))
print("Reranker:", type(resources.reranker))