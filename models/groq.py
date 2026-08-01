from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# LLM model for multi query generation
query_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key = api_key
)

# LLM model for final answers
answer_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key = api_key
)