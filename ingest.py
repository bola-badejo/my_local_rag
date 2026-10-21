# ingest.py
"""
Run this whenever you have new documents. Point it at files,
it eats them and stores the chunks in chroma_db/.

Usage:
  python ingest.py                    # ingest everything in docs/
  python ingest.py docs/manual.pdf    # ingest a specific file
  python ingest.py https://foo.com    # ingest a URL
  python ingest.py --reset            # wipe DB, re-ingest everything in docs/
"""

import os
import sys
from config import DOCS_DIR
from loaders import load_any
from chunker import chunk_text
from vectorstore import add_chunks, reset, get_collection
from config import CHUNK_SIZE, CHUNK_OVERLAP


def ingest_one(source):
    """One file or URL → load → chunk → store."""
    print(f"📄 {source}")

    try:
        text = load_any(source)
    except Exception as e:
        print(f"   ❌ Couldn't load: {e}")
        return

    if not text.strip():
        print("   ⚠️  No text extracted (scanned PDF? empty file?)")
        return

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    n = add_chunks(chunks, source)
    print(f"   ✅ {n} chunks stored")


def find_all_docs():
    """
    Walks docs/ recursively and returns every supported file path.
    This is what makes `python ingest.py` with no args work —
    it just ingests EVERYTHING in the docs folder.
    """
    supported = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md"}
    found = []
    for root, _, files in os.walk(DOCS_DIR):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in supported:
                found.append(os.path.join(root, name))
    return found


if __name__ == "__main__":
    args = sys.argv[1:]

    # --reset wipes the DB before ingesting. Useful when you've
    # changed CHUNK_SIZE and don't want mixed-size chunks in there.
    if "--reset" in args:
        reset()
        args = [a for a in args if a != "--reset"]

    # No args = ingest every file in docs/. With args = ingest just those.
    sources = args if args else find_all_docs()

    if not sources:
        print(f"No files found in {DOCS_DIR}. Put some documents there first!")
        sys.exit(0)

    print(f"Ingesting {len(sources)} source(s)...\n")
    for src in sources:
        ingest_one(src)

    print(f"\n🎉 Done. You now have {get_collection().count()} chunks in the database.")
    print("   Ask questions with:  python qa.py")