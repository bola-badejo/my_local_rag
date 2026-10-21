# vectorstore.py
"""
The memory system. Chroma stores vectors in a folder on disk.
Ollama turns text into vectors — locally, on your CPU/GPU.

Key concept: an embedding is just a list of ~768 numbers that
represents the MEANING of a piece of text. Similar meanings
produce similar numbers. That's the whole trick behind
semantic search.
"""

import ollama
import chromadb
from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, TOP_K


def get_collection():
    """
    Opens the DB. If chroma_db/ doesn't exist yet, creates it.
    If the collection doesn't exist yet, creates that too.
    So the FIRST call to this function is what actually sets
    up the database on your disk.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine = good for text similarity
    )


def embed(text):
    """
    One string → one vector (list of floats).
    ollama.embeddings runs the local nomic-embed-text model.
    No network. No API key.
    """
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def add_chunks(chunks, source):
    """
    Stores a batch of chunks from one document.

    Each chunk gets:
      - its embedding (the vector)
      - its raw text (so we can show it to the LLM later)
      - metadata (at minimum: which file it came from)
      - a unique ID (required by Chroma)

    IDs are made from source+index so re-ingesting the same file
    overwrites cleanly instead of duplicating. That's a nice
    safety net — you can re-run ingest.py without worrying
    about doubles.
    """
    if not chunks:
        return 0

    collection = get_collection()
    embeddings = [embed(c) for c in chunks]
    ids = [f"{source}::{i}" for i in range(len(chunks))]
    metadatas = [{"source": source} for _ in chunks]

    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
    )
    return len(chunks)


def search(question, k=TOP_K):
    """
    The core retrieval step. Given a question:
      1. Embed it (same way we embedded the chunks)
      2. Ask Chroma for the k nearest vectors
      3. Return the chunks + their sources

    'Nearest' here means closest in cosine distance — i.e.
    the chunks whose MEANING is most similar to the question.
    It's not keyword matching. It finds "password reset"
    when you ask "how do I recover my account" because those
    sentences have similar semantic embeddings.
    """
    collection = get_collection()
    if collection.count() == 0:
        return [], []

    q_vec = embed(question)
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=k,
    )

    # Chroma returns lists-of-lists (you can query multiple vectors
    # at once). We only sent one, so unwrap index [0].
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    return docs, metas


def reset():
    """
    Nuclear option. Deletes the entire collection.
    Useful when you've changed chunk settings and want to re-ingest
    from scratch. Does NOT touch your docs/ folder — those stay safe.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("💥 Vector database wiped. Re-run ingest.py to rebuild.")
    except Exception as e:
        print(f"Nothing to wipe ({e})")