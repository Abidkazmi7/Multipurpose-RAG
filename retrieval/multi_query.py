from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import dumps, loads

# Splits generated queries into a clean list of strings
def parse_queries(x: str):
    return [
        q.strip()
        for q in x.split("\n")
        if q.strip()
    ]

# Create multiple queries for expanded context
def create_multi_query(llm):
    template = """You are an AI language model assistant. Your task is to generate three 
    alternative versions that preserve the exact intent of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Generate only the
    questions and do not say anything else. Original question: {question}"""

    prompt_perspectives = PromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives
        | llm
        | StrOutputParser()
        | parse_queries
    )

    return generate_queries

# Get unique documents
def get_unique_union(documents: list[list]):
    seen = set()
    unique_docs = []

    for sublist in documents:
        for doc in sublist:
            child_id = doc.metadata["child_id"]

            if child_id not in seen:
                seen.add(child_id)
                unique_docs.append(doc)

    return unique_docs