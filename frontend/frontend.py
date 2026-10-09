import streamlit as st
import uuid
import os

from langchain_core.messages import HumanMessage

from backend.chat import Chat
from backend.resources import resources
from backend.database import create_thread, get_thread, retrieve_all_threads

# -------------------- UTILITY FUNCTIONS --------------------

def create_thread_id():
    return str(uuid.uuid4())

def reset_chat(modality, source):
    thread_id = create_thread_id()

    # Add thread details to database
    create_thread(
        thread_id=thread_id,
        modality=modality,
        source=source
    )

    st.session_state.thread_id = thread_id     
    st.session_state.modality = modality
    st.session_state.source = source
    st.session_state.message_history = []

    # Add thread_id to session state
    add_thread(thread_id)

def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

# Load conversation history for a particular thread id
def load_conversation(thread_id):
    state =  st.session_state.chat.graph.get_state(
        config={'configurable': {'thread_id': thread_id}})

    return state.values.get('messages', [])

# Saving user's uploaded pdf
def save_uploaded_file(uploaded_file):
    upload_dir = os.path.join("data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

# -------------------- SESSION SETUP --------------------

if 'chat_threads' not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

# Initialize chatbot once
if "chat" not in st.session_state:
    st.session_state.chat = Chat(resources)

# Frontend message history for displaying chat
if "message_history" not in st.session_state:
    st.session_state.message_history = []

# -------------------- SIDEBAR UI   --------------------

st.sidebar.title('Multi-purpose Chatbot')

modality = st.sidebar.selectbox(
    "Select modality",
    ["youtube", "document", "web"]
)

# Source input based on selected modality
if modality == "document":
    uploaded_file = st.sidebar.file_uploader(
        "Upload document",
        type=["pdf", "docx", "txt"]
    )

    source = uploaded_file

    if uploaded_file is not None:
        source = save_uploaded_file(uploaded_file)

elif modality == "youtube":
    source = st.sidebar.text_input(
        "YouTube URL"
    )

else:
    source = st.sidebar.text_input(
        "Webpage URL"
    )

if st.sidebar.button("Start New Chat"):
    if source is None or source == "":
        st.sidebar.warning("Please provide a source.")

    else:
        reset_chat(modality, source)

    # Initialize Chat for the newly created thread
    st.session_state.chat.initialize(
        modality=modality,
        source=source
    )

st.sidebar.header('My Conversations')

for thread_id in st.session_state.chat_threads[::-1]:
    if st.sidebar.button(thread_id):
        thread = get_thread(thread_id)

        st.session_state.thread_id = thread_id
        st.session_state.modality = thread["modality"]
        st.session_state.source = thread["source"]

        # Reinitialize Chat for this thread's modality/source
        st.session_state.chat.initialize(
            modality=st.session_state.modality,
            source=st.session_state.source
        )

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
user_input = st.chat_input(
    "Ask something about the document...",
    disabled="thread_id" not in st.session_state
)

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