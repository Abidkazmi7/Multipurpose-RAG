from langchain_community.vectorstores import Chroma

# Chroma vectorspace
def vector_database(chunks, model):
    vectorstore = Chroma.from_documents(documents = chunks, embedding = model)

    return vectorstore