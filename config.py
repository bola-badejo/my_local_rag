# config.py
"""
Every tunable value lives here. Change once, applies everywhere.

If you're asking "why is the answer bad?", this file is
the first place you should look. Chunk size and top_k
are the two biggest quality levers in RAG.
"""

import os

# Where things live
# We compute absolute paths from THIS file's location, so it doesn't matter
# what directory you run the scripts from. This one trick prevents
# 90% of "file not found" errors people hit.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

#  Models
# Embeddings run locally inside this process via sentence-transformers.
# First startup downloads ~90MB from HuggingFace and caches it.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chat model is hosted (Gemini). Needs GEMINI_API_KEY in st.secrets or env.
# Flash = fast + free tier. Swap to "gemini-2.5-pro" for best quality.
CHAT_MODEL = "gemini-2.5-flash"

#  Chunking 
# 800 chars ≈ 200 tokens. Overlap means consecutive chunks share
# some text so ideas don't get chopped mid-sentence.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

#  Retrieval 
# How many chunks to fetch when answering a question.
# 4 is a good default. Bump to 6-8 if answers feel thin.
TOP_K = 4

#  Chroma collection name 
# All chunks live in one "table" — this is its name.
COLLECTION_NAME = "my_docs"