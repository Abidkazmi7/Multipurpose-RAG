from langchain_community.vectorstores import Chroma

# Chroma vectorspace
def chroma_retriever(chunks, model):
    # Create vectorstore
    vectorstore = Chroma.from_documents(documents = chunks, embedding = model)

    # Create retriever object
    retriever = vectorstore.as_retriever()

    return retriever