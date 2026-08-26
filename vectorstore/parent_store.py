from langchain_core.stores import InMemoryStore

def build_parent_store(parents):
    store = InMemoryStore()

    # Store each parent document using its parent_id
    store.mset([
        (parent.metadata["parent_id"], parent) for parent in parents
    ])

    return store