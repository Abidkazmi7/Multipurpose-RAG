from langchain_groq import ChatGroq

# LLM model for multi query generation
query_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key = "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"
)

# LLM model for final answers
answer_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key = "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"
)