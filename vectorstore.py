# vectorstore.py
"""
The memory system. Chroma stores vectors in a folder on disk.
sentence-transformers turns text into vectors — locally, inside
this Python process. No API key, no network, free forever.

Key concept: an embedding is just a list of ~384 numbers that
represents the MEANING of a piece of text. Similar meanings
produce similar numbers. That's the whole trick behind
semantic search.
"""

# Streamlit Cloud ships an old sqlite3; swap it for pysqlite3 before
# importing chromadb. No-op locally if pysqlite3 isn't installed.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, TOP_K


# Load the embedding model once per process. First call downloads
# ~90MB from HuggingFace and caches it; subsequent calls are instant.
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_collection():
    """
    Opens the DB. If chroma_db/ doesn't exist yet, creates it.
    If the collection doesn't exist yet, creates that too.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed(text):
    """One string → one vector (list of floats)."""
    vec = _get_model().encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts):
    """Many strings → many vectors. Much faster than calling embed() in a loop."""
    vecs = _get_model().encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32
    )
    return [v.tolist() for v in vecs]


def add_chunks(chunks, source):
    """
    Stores a batch of chunks from one document.

    Each chunk gets:
      - its embedding (the vector)
      - its raw text (so we can show it to the LLM later)
      - metadata (at minimum: which file it came from)
      - a unique ID (required by Chroma)

    IDs are made from source+index so re-ingesting the same file
    overwrites cleanly instead of duplicating.
    """
    if not chunks:
        return 0

    collection = get_collection()
    embeddings = embed_batch(chunks)
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
    """
    collection = get_collection()
    if collection.count() == 0:
        return [], []

    q_vec = embed(question)
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=k,
    )

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
        print("Vector database wiped. Re-run ingest.py to rebuild.")
    except Exception as e:
        print(f"Nothing to wipe ({e})")
