from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from generation.pipeline import retrieval_chain, youtube_retriever, doc_retriever
from operator import itemgetter
from models.groq import answer_llm
from retrieval.reranker import rerank_docs
from retrieval.fetch_parent_docs import get_parents

# Link Langsmith
from dotenv import load_dotenv
load_dotenv()

pdf_path = "E:\Artificial Intelligence\RAG\Multi-purpose RAG System\data\Prospectus.pdf"
# url = "https://www.youtube.com/watch?v=FLcrvMfHUJM"

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

    # Get retriever object & parent document store
    retriever, parent_store = doc_retriever(pdf_path)

    # Retrieve unique docs
    chain = retrieval_chain(retriever)

    prompt = PromptTemplate.from_template(doc_template)

    # Rerank retrieved docs & fetch their corresponding parent document
    reranked_retrieval = (
        {
            "docs" : chain,
            "question" : itemgetter("question")
        }
        | RunnableLambda(rerank_docs)
        | RunnableLambda(lambda docs: get_parents(docs, parent_store))
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

question = "What are the 5th semester courses for Bachelor's in Mechanical Engineering?"
answer = final_prompt(question)
print(f"ANSWER: {answer}")