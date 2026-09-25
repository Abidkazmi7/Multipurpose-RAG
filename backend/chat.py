from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from rag.generation.pipeline import(
    retrieval_chain,
    doc_retriever,
    youtube_retriever,
    webpage_retriever
)

from rag.retrieval.reranker import rerank_docs
from rag.retrieval.fetch_parent_docs import get_parents
from rag.citations.citation import extract_citations

from backend.graph import create_chat_graph

class Chat:
    def __init__(self, resources):
        self.resources = resources

        self.modality = None
        self.retriever = None
        self.retrieval_chain = None
        self.parent_store = None
        self.extract_citations = extract_citations
        self.graph = None

    def initialize(self, modality, source):
        self.modality = modality

        if modality == "document":
            self.retriever, self.parent_store = doc_retriever(
                source, self.resources.embedding_model
            )

        elif modality == "youtube":
            self.retriever = youtube_retriever(
                source, self.resources.embedding_model
            )

        elif modality == "web":
            self.retriever = webpage_retriever(
                source, self.resources.embedding_model
            )

        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # Create RAG workflow
        self.build_retrieval_chain()

        # Answer prompts
        self.build_answer_prompt()

        # Initialize graph
        self.graph = create_chat_graph(self)

    def ask(self, question, thread_id):
        config = {
            "configurable": {"thread_id": thread_id}
        }

        result = self.graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            },
            config = config
        )

        return result
    
    def build_retrieval_chain(self):
        chain = retrieval_chain(self.retriever)

        self.retrieval_chain = (
            {
                "docs" : chain,
                "question" : itemgetter("question")
            }
            | RunnableLambda(
                lambda inputs: rerank_docs(inputs, self.resources.reranker)
            )
            | RunnableLambda(
                    lambda docs: get_parents(docs, self.parent_store)
                    if self.parent_store is not None
                    else docs 
                )
        )

    def build_answer_prompt(self):
        self.general_answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a helpful conversational assistant.

                Answer the user's latest question naturally and accurately
                using your general knowledge.

                Do not claim that the answer comes from the user's knowledge base.
                """
            ),
            MessagesPlaceholder(variable_name="messages")
        ])

        if self.modality == "document":
            self.answer_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """
                    Answer the user's question based strictly on the provided context.

                    If the context is insufficient, just context could not be found in the knowledge base.

                    Retrieved context:
                    {context}
                    """
                ),
                MessagesPlaceholder(variable_name="messages")
            ])

        elif self.modality == "youtube":
            self.answer_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """
                    You are a helpful assistant.

                    Answer ONLY from the provided transcript context.

                    If the context is insufficient, just say context could not be found in the knowledge base.

                    Transcript context:
                    {context}
                    """
                ),
                MessagesPlaceholder(variable_name="messages")
            ])

        elif self.modality == "web":
            self.answer_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """
                    You are a helpful assistant.

                    Answer the question ONLY using the provided webpage context.

                    Do not use outside knowledge or make assumptions.

                    If the context is insufficient to answer the question, just say context could not be found in the knowledge base.

                    Webpage context:
                    {context}
                    """
                ),
                MessagesPlaceholder(variable_name="messages")
            ])