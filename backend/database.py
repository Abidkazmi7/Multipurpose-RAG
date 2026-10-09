import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn=conn)

# Insert table for thread metadata (source & modality)
conn.execute("""
    CREATE TABLE IF NOT EXISTS threads (
        thread_id TEXT PRIMARY KEY,
        modality TEXT NOT NULL,
        source TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

conn.commit()

def create_thread(thread_id, modality, source):
    conn.execute(
        """
        INSERT INTO threads (thread_id, modality, source)
        VALUES (?, ?, ?)
        """,
        (thread_id, modality, source)
    )
    conn.commit()

def get_thread(thread_id):
    cursor = conn.execute(
        """
        SELECT thread_id, modality, source
        FROM threads
        WHERE thread_id = ?
        """,
        (thread_id,)
    )

    row = cursor.fetchone()

    if row is None:
        return None

    return{
        "thread_id": row[0],
        "modality": row[1],
        "source": row[2]
    }

# List of all threads in the database
def retrieve_all_threads():
    cursor = conn.execute(
        """
        SELECT thread_id
        FROM threads
        ORDER BY created_at DESC
        """
    )

    return [row[0] for row in cursor.fetchall()]