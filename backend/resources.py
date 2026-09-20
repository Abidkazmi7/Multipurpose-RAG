from rag.models.embedding_models import(
    huggingface_model,
    crossencoder_model
)

from rag.models.language_models import(
    query_llm,
    answer_llm
)

class Resources:
    def __init__(self):
        print("Initializing LLM models...")
        self.multiquery_model = query_llm
        self.answer_model = answer_llm

        print("Loading embedding model...")
        self.embedding_model = huggingface_model()

        print("Loading reranking model...")
        self.reranker = crossencoder_model()

        print("Models successfully loaded.")

resources = Resources()