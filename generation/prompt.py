from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from generation.pipeline import retrieval_chain, youtube_retriever, doc_retriever
from operator import itemgetter
from models.groq import answer_llm
from retrieval.reranker import rerank

# Link Langsmith
from dotenv import load_dotenv
load_dotenv()

pdf_path = "data\Prospectus.pdf"
# url = "https://www.youtube.com/watch?v=FLcrvMfHUJM"

def rerank_docs(inputs):
    return rerank(
        query = inputs["question"],
        docs = inputs["docs"]
    )

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

    # Retrieve unique docs
    chain = retrieval_chain(doc_retriever(pdf_path))

    prompt = PromptTemplate.from_template(doc_template)

    # Rerank retrieved docs
    reranked_retrieval = (
        {
            "docs" : chain,
            "question" : itemgetter("question")
        }
        | RunnableLambda(rerank_docs)
    )

    # Final RAG chain
    final_rag_chain = (
        {"context": reranked_retrieval,
        "question": itemgetter("question")}
        | prompt
        | answer_llm
        | StrOutputParser()
    )

    answer = final_rag_chain.invoke({"question": question})

    return answer

question = "What do the students have to say about the Bachelors in Chemical Engineering program?"
answer = final_prompt(question)
print(f"ANSWER: {answer}")