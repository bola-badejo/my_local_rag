# qa.py
"""
The Q&A loop, all local.

Two steps:
  1. RETRIEVE — find chunks whose embeddings match the question
  2. GENERATE — hand those chunks + the question to llama3.2 and
                say "answer using ONLY this context"

That second part matters. Without it, the model will answer from
its training data and you'll get confident-sounding garbage.
"""

import ollama
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


def answer_question(question, k=TOP_K):
    """
    Returns {"answer": str, "sources": list[str]}.

    Note the k parameter — you can bump it per-question if you
    want more context for a hard question without changing the
    global config.
    """
    #  Retrieve 
    docs, metas = search(question, k=k)

    if not docs:
        return {
            "answer": "No documents are indexed yet. Run `python ingest.py` first.",
            "sources": [],
        }

    # Build the context string with source tags on each chunk.
    # The tags let the LLM say "according to manual.pdf" naturally.
    context = "\n\n\n\n".join(
        f"[Source: {m.get('source', 'unknown')}]\n{d}"
        for d, m in zip(docs, metas)
    )
    sources = sorted({m.get("source", "unknown") for m in metas})

    #  Generate 
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ],
        options={"temperature": 0},  # 0 = deterministic, factual
    )

    return {
        "answer": response["message"]["content"],
        "sources": sources,
    }


if __name__ == "__main__":
    print("💬 Local RAG bot ready. Type 'quit' to exit.\n")
    while True:
        try:
            q = input("❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = answer_question(q)
        print(f"\n💬 {result['answer']}")
        if result["sources"]:
            print(f"📚 {', '.join(result['sources'])}\n")
        else:
            print()