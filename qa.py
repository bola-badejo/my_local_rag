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


def _rank_candidates():
    """
    Ask Gemini which models this key can actually call, filter to realistic
    chat models, and return them ranked best-first.
    """
    try:
        listed = list(_get_client().models.list())
    except Exception as e:
        return [], f"could not list models: {type(e).__name__}: {e}"

    # Keywords that disqualify a model from being tried as a chat backend.
    # These are experimental / specialized / expensive-and-no-free-quota variants.
    BLOCK = (
        "embed", "embedding", "tts", "image-gen", "imagegen",
        "live", "native-audio", "aqa", "learnlm",
        "deep-research", "research-pro",
        "thinking", "reasoning",
        "veo", "imagen",
    )
    SOFT_PENALTY = ("preview", "exp", "experimental", "-dev", "-latest")

    import re

    callable_models = []
    for m in listed:
        name = (getattr(m, "name", "") or "").replace("models/", "")
        if not name:
            continue
        methods = (
            getattr(m, "supported_actions", None)
            or getattr(m, "supported_generation_methods", None)
            or []
        )
        if methods and "generateContent" not in methods:
            continue
        if any(bad in name for bad in BLOCK):
            continue
        callable_models.append(name)

    if not callable_models:
        return [], f"no usable chat models visible to this key (saw {len(listed)} total)"

    def score(name):
        s = 0
        if "flash" in name and "lite" in name: s += 120   # cheapest / most free quota
        elif "flash" in name: s += 100
        elif "pro" in name: s += 60
        # version number bump
        m = re.search(r"(\d+(?:\.\d+)?)", name)
        if m:
            try: s += float(m.group(1)) * 10
            except ValueError: pass
        for p in SOFT_PENALTY:
            if p in name: s -= 25
        # bias away from dated snapshots like "-11-2025" when a cleaner name exists
        if re.search(r"-\d{1,2}-\d{4}$", name): s -= 10
        return s

    callable_models.sort(key=score, reverse=True)
    return callable_models, f"{len(callable_models)} candidates"


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

    # If we already found a working model this session, go straight to it.
    if _resolved_model:
        try:
            response = _call_model(_resolved_model, question, context)
            return {"answer": response.text, "sources": sources}
        except Exception:
            # Cached model stopped working (quota?) — re-detect below.
            _resolved_model = None

    # Build the try-order: configured model first, then ranked list from the API.
    attempt_errors = []
    try_order = [CHAT_MODEL]
    ranked, note = _rank_candidates()
    for name in ranked:
        if name not in try_order:
            try_order.append(name)

    for candidate in try_order[:6]:  # cap retries so we don't burn the clock
        try:
            response = _call_model(candidate, question, context)
            _resolved_model = candidate
            suffix = ""
            if candidate != CHAT_MODEL:
                suffix = f"\n\n_(auto-selected model: `{candidate}`)_"
            return {"answer": response.text + suffix, "sources": sources}
        except Exception as e:
            attempt_errors.append((candidate, type(e).__name__, str(e)[:240]))

    # Nothing worked — show the user which candidates we tried and what went wrong.
    lines = [f"**No model this key can call succeeded.** ({note})\n"]
    for name, etype, msg in attempt_errors:
        lines.append(f"- `{name}` → `{etype}`: {msg}")
    return {"answer": "\n".join(lines), "sources": sources}


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
