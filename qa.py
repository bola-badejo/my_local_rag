# qa.py
"""
The Q&A loop.

Two steps:
  1. RETRIEVE — find chunks whose embeddings match the question
  2. GENERATE — hand those chunks + the question to Gemini and
                say "answer using ONLY this context"
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


# Resolved model name picked at runtime. None until the first successful call
# (or the first fallback probe). Cached process-wide so we only auto-detect once.
_resolved_model = None


def _pick_available_model():
    """
    Ask Gemini which models this key can actually call. Prefer the newest
    Flash variant (cheap + fast enough for RAG), then Pro, then anything
    that supports generateContent.
    """
    try:
        listed = list(_get_client().models.list())
    except Exception as e:
        return None, f"could not list models: {type(e).__name__}: {e}"

    # Filter to models that support generateContent
    callable_models = []
    for m in listed:
        name = getattr(m, "name", "") or ""
        methods = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
        if "generateContent" in methods or not methods:
            callable_models.append(name.replace("models/", ""))

    if not callable_models:
        return None, f"no generative models visible to this key (saw {len(listed)} total)"

    def score(name):
        s = 0
        if "flash" in name: s += 100
        if "pro" in name: s += 50
        # Prefer numbered stable releases over "latest" aliases
        import re
        m = re.search(r"(\d+(?:\.\d+)?)", name)
        if m:
            try: s += float(m.group(1)) * 10
            except ValueError: pass
        if "exp" in name or "preview" in name: s -= 30
        if "tts" in name or "image" in name or "embed" in name: s -= 1000
        return s

    callable_models.sort(key=score, reverse=True)
    return callable_models[0], f"picked {callable_models[0]} from {len(callable_models)} candidates"


def _call_model(model, question, context):
    return _get_client().models.generate_content(
        model=model,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.format(context=context),
            temperature=0,
        ),
    )


def answer_question(question, k=TOP_K):
    """Returns {"answer": str, "sources": list[str]}."""
    global _resolved_model

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

    model_to_try = _resolved_model or CHAT_MODEL

    try:
        response = _call_model(model_to_try, question, context)
        _resolved_model = model_to_try
        return {"answer": response.text, "sources": sources}

    except Exception as first_err:
        # First attempt failed — try to auto-detect a working model and retry.
        picked, picked_note = _pick_available_model()
        if picked and picked != model_to_try:
            try:
                response = _call_model(picked, question, context)
                _resolved_model = picked
                return {
                    "answer": response.text + f"\n\n_(auto-selected model: `{picked}`)_",
                    "sources": sources,
                }
            except Exception as second_err:
                return {
                    "answer": (
                        f"**Both the configured and the auto-selected model failed.**\n\n"
                        f"Configured: `{model_to_try}` → `{type(first_err).__name__}`: {first_err}\n\n"
                        f"Auto-picked: `{picked}` → `{type(second_err).__name__}`: {second_err}"
                    ),
                    "sources": sources,
                }

        return {
            "answer": (
                f"**Model call failed and auto-detection could not help.**\n\n"
                f"Configured: `{model_to_try}`\n"
                f"Error: `{type(first_err).__name__}`: {first_err}\n\n"
                f"Auto-detection: {picked_note}"
            ),
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
