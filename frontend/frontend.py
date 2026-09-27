import streamlit as st
import uuid
from langchain_core.messages import HumanMessage

from backend.chat import Chat
from backend.resources import resources
from backend.database import retrieve_all_threads

# -------------------- UTILITY FUNCTIONS --------------------

def create_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = create_thread_id()
    st.session_state.thread_id = thread_id     
    add_thread(st.session_state.thread_id)
    st.session_state.message_history = []

def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

# Load conversation history for a particular thread id
def load_conversation(thread_id):
    state =  st.session_state.chat.graph.get_state(
        config={'configurable': {'thread_id': thread_id}})

    return state.values.get('messages', [])


# -------------------- SESSION SETUP --------------------

# Initialize chatbot once
if "chat" not in st.session_state:
    st.session_state.chat = Chat(resources)

    st.session_state.chat.initialize(
        modality="web",
        source="https://huggingface.co/learn/llm-course/en/chapter1/8"
    )

# Frontend message history for displaying chat
if "message_history" not in st.session_state:
    st.session_state.message_history = []

# Create random thread_id when session starts
if "thread_id" not in st.session_state:
    st.session_state.thread_id = create_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

add_thread(st.session_state.thread_id)

# -------------------- SIDEBAR UI   --------------------

st.sidebar.title('Multipurpose Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state.chat_threads[::-1]:
    if st.sidebar.button(thread_id):
        st.session_state.thread_id = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'

            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state.message_history = temp_messages

# -------------------- MAIN UI --------------------

# Display previous messages
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Ask something about the document...")

if user_input:
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in st.session_state.chat.stream(
                user_input,
                st.session_state.thread_id
            )

            # Stream messages only from generation nodes
            if metadata.get("langgraph_node") in [
                "generate_using_retrieval",
                "generate_using_llm"
            ]
        )

    st.session_state.message_history.append({
        "role": "assistant",
        "content": ai_message
    })