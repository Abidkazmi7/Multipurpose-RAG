from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from backend.resources import query_llm

# Schema for deciding whether question needs to be contextualized
class QueryContextualize(TypedDict):
    should_rewrite: bool
    question: str

# Schema for LLM deciding whether or not retrieval is required
class ContextDecision(TypedDict):
    retrieval_required: bool

class ChatState(TypedDict):
    # Conversation history
    messages: Annotated[list, add_messages]

    # Context for current question
    context: list

    # Rewritten query to incorporate context
    contextualized_question: str

    # Citations for current question's answer
    citations: list

    # Source of answer (User documents/LLM knowledge)
    answer_src: str

def create_chat_graph(chat):
    graph = StateGraph(ChatState)

    def contextualize_node(state: ChatState):
        conversation = state["messages"][:-1]
        question = state["messages"][-1].content

        prompt = f"""
        You are a query contextualization component in a conversational RAG system.

        Your task is to determine whether the user's latest question depends on
        previous conversation.

        Rules:

        1. If the question is already self-contained, do NOT rewrite it.
        Return the exact original question and set should_rewrite to false.

        2. If the question depends on previous conversation, rewrite it into a
        self-contained question and set should_rewrite to true.

        3. Greetings, thanks, farewells, acknowledgements, and similar messages
        must NOT be rewritten. Return them exactly as they were provided.

        4. Do not change the wording of a question unless previous conversation
        is required to understand it.

        Conversation:
        {conversation}

        Latest question:
        {question}
        """

        contextualizer = query_llm.with_structured_output(
            QueryContextualize,
            method="json_schema"
        )

        result = contextualizer.invoke(prompt)

        return {
            "contextualized_question": result["question"]
        }

    # Retrieval node
    def retrieve_node(state: ChatState):
        docs = chat.retrieval_chain.invoke({
            "question": state["contextualized_question"]
        })

        citations = chat.extract_citations(docs)

        return {
            "context": docs,
            "citations": citations
        }

    # Decide whether to answer from retrieval or LLM knowledge
    def judge_node(state: ChatState):
        question = state["contextualized_question"]
        context = state["context"]

        prompt = """
        You are a context-relevance judge for a RAG system.

        Determine whether the provided context contains enough information
        to answer the user's question accurately.

        Return true ONLY if the question can be answered using the provided
        context. Return false if the context is missing, irrelevant, or
        insufficient to answer the question.

        Question:
        {question}

        Context:
        {context}
        """

        judge = query_llm.with_structured_output(ContextDecision, method="json_schema")

        result = judge.invoke(
            prompt.format(
                question=question,
                context=context
            )
        )

        return{
            "retrieval_required": result["retrieval_required"]
        }

    # Generate using retrieved context
    def generate_using_retrieval(state: ChatState):
        response = (
            chat.answer_prompt
            | chat.resources.answer_model
        ).invoke({
            "messages": state["messages"],
            "context": state["context"]
        })

        return {
            "messages": [response],
            "answer_source": "knowledge_base"
        }

    # Generate using LLM's knowledge
    def generate_using_llm(state: ChatState):
        response = (
            chat.general_answer_prompt
            | chat.resources.answer_model
        ).invoke({
            "messages": state["messages"]
        })

        return{
            "messages": response,
            "answer_source": "llm"
        }

    graph.add_node("contextualize_query", contextualize_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("judge", judge_node)
    graph.add_node("generate_using_retrieval", generate_using_retrieval)
    graph.add_node("generate_using_llm", generate_using_llm)

    graph.add_edge(START, "contextualize_query")
    graph.add_edge("contextualize_query", "retrieve")
    graph.add_edge("retrieve", "judge")

    graph.add_conditional_edges(
        "judge",
        lambda state: (
            "generate_using_retrieval"
            if state["retrieval_required"]
            else "generate_using_llm"
        )
    )

    graph.add_edge("generate_using_retrieval", END)
    graph.add_edge("generate_using_llm", END)

    checkpointer = MemorySaver()

    return graph.compile(
        checkpointer = checkpointer
    )