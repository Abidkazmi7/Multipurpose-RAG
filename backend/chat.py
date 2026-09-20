from operator import itemgetter

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

from rag.generation.pipeline import(
    retrieval_chain,
    doc_retriever,
    youtube_retriever,
    webpage_retriever
)

from rag.retrieval.reranker import rerank_docs
from rag.retrieval.fetch_parent_docs import get_parents
from rag.citations.citation import extract_citations

class Chat:
    def __init__(self, resources):
        self.resources = resources

        self.modality = None
        self.retriever = None
        self.parent_store = None
        self.rag_chain = None

        self.conversation_history = []

    def ask(self, question):
        result = self.rag_chain.invoke({
            "question": question
        })

        self.conversation_history.append({
            "question": question,
            "answer": result["answer"]
        })

        return result

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

        self._build_rag_chain()

    def _build_rag_chain(self):
        chain = retrieval_chain(self.retriever)

        if self.modality == "document":
            prompt_template = """
            Answer the following question based strictly on this context.
            If the context is insufficient, just say you don't know.

            {context}

            Question: {question}
            """

        elif self.modality == "youtube":
            prompt_template = """
            You are a helpful assistant.

            Answer ONLY from the provided transcript context.

            If the context is insufficient, just say you don't know.

            {context}

            Question: {question}
            """

        elif self.modality == "web":
            prompt_template = """
            You are a helpful assistant.

            Answer the question ONLY using the provided webpage context.

            Do not use outside knowledge or make assumptions.

            If the context is insufficient to answer the question, just say you don't know.

            Webpage context:

            {context}

            Question: {question}
            """

        prompt = PromptTemplate.from_template(prompt_template)

        reranked_retrieval = (
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

        self.rag_chain = (
            {
                "context": reranked_retrieval,
                "question": itemgetter("question")
            }
            | RunnableParallel(
                answer = (
                    {
                        "context": itemgetter("context"),
                        "question": itemgetter("question")
                    }
                    | prompt
                    | self.resources.answer_model
                    | StrOutputParser()
                ),
                citations = (
                    itemgetter("context")
                    | RunnableLambda(extract_citations)
                )
            )
        )