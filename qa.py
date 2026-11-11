# qa.py
"""
The Q&A loop.

Two steps:
  1. RETRIEVE — find chunks whose embeddings match the question
  2. GENERATE — hand those chunks + the question to Gemini and
                say "answer using ONLY this context"

That second part matters. Without it, the model will answer from
its training data and you'll get confident-sounding garbage.
"""

import os
from google import genai
from google.genai import types

from config import CHAT_MODEL, TOP_K
from vectorstore import search


SYSTEM_PROMPT = """You are a precise assistant that answers questions \
using ONLY the provided context.

Rules:
- If the answer isn't in the context, say "I don't have enough information in the documents to answer that." Do NOT guess.
- When you use information from a source, mention the source filename.
- Be concise. Two or three sentences is usually enough.
- If the context contains partial info, say what you do know and what's missing.

Context:
{context}"""


def _get_api_key():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to .streamlit/secrets.toml "
            "or export it as an environment variable."
        )


_client = None

def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=_get_api_key())
    return _client


def answer_question(question, k=TOP_K):
    """Returns {"answer": str, "sources": list[str]}."""
    docs, metas = search(question, k=k)

    if not docs:
        return {
            "answer": "No documents are indexed yet. Upload something in the sidebar first.",
            "sources": [],
        }

    context = "\n\n\n\n".join(
        f"[Source: {m.get('source', 'unknown')}]\n{d}"
        for d, m in zip(docs, metas)
    )
    sources = sorted({m.get("source", "unknown") for m in metas})

    response = _get_client().models.generate_content(
        model=CHAT_MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.format(context=context),
            temperature=0,
        ),
    )

    return {
        "answer": response.text,
        "sources": sources,
    }


if __name__ == "__main__":
    print("RAG bot ready. Type 'quit' to exit.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = answer_question(q)
        print(f"\n{result['answer']}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}\n")
        else:
            print()
