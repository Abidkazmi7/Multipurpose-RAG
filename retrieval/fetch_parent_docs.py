def get_parents(child_docs, parent_store):
    parent_ids = []

    for doc in child_docs:
        parent_id = doc.metadata["parent_id"]

        # Avoid storing same parent chunk twice
        if parent_id not in parent_ids:
            parent_ids.append(parent_id)

    parents = parent_store.mget(parent_ids)

    return [parent for parent in parents if parent is not None]