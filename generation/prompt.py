from operator import itemgetter

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

from generation.pipeline import retrieval_chain, youtube_retriever, doc_retriever, webpage_retriever

from models.language_models import answer_llm

from retrieval.reranker import rerank_docs
from retrieval.fetch_parent_docs import get_parents

from citations.citation import extract_citations

# Link Langsmith
from dotenv import load_dotenv
load_dotenv()

# pdf_path = "e:\Artificial Intelligence\RAG\Research Paper RAG\knowledge_base\documents\LLM_Improving Language Understanding by Generative Pre-Training.pdf"
# yt_url = "https://youtu.be/HQA7fxZ-_r0?si=VNdQiZfX-IVWhWdt"
web_url = "https://www.ibm.com/think/insights/10-ai-dangers-and-risks-and-how-to-manage-them"

# Final prompt fed to LLM
def final_prompt(question):
    doc_template = """
        Answer the following question based strictly on this context:

        {context}

        Question: {question}
    """
    yt_template = """
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}

        Question: {question}
    """
    web_template = """
        You are a helpful assistant.
        Answer the question ONLY using the provided webpage context.
        Do not use outside knowledge or make assumptions.
        If the context is insufficient to answer the question, just say you don't know.

        Webpage context:
        {context}

        Question: {question}
    """

    # Get retriever object & parent document store

    parent_store = None
    # retriever, parent_store = doc_retriever(pdf_path)
    # retriever = youtube_retriever(yt_url)
    retriever = webpage_retriever(web_url)

    # Retrieve unique docs
    chain = retrieval_chain(retriever)

    prompt = PromptTemplate.from_template(web_template)

    # Rerank retrieved docs & fetch their corresponding parent document
    reranked_retrieval = (
        {
            "docs" : chain,
            "question" : itemgetter("question")
        }
        | RunnableLambda(rerank_docs)
        | RunnableLambda(
                lambda docs: get_parents(docs, parent_store)
                if parent_store is not None
                else docs 
            )
    )

    # Final RAG chain
    final_rag_chain = (
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
                | answer_llm
                | StrOutputParser()
            ),
            citations = (
                itemgetter("context")
                | RunnableLambda(extract_citations)
            )
        )
    )

    result = final_rag_chain.invoke({"question": question})

    return result

question = "How does AI affect the job market and what can we do to be safe?"
result = final_prompt(question)

print(f"ANSWER: {result["answer"]}")
print(f"CITATIONS: {result["citations"]}")