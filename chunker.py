# chunker.py
"""
Splitting a long text into overlapping chunks.

I'm doing this by hand instead of using LangChain because
it's 15 lines, and you can see exactly what's happening.
No magic imports.

Overlap is important. Without it, you can split right through
the middle of "The password reset process is:" and lose the
actual steps — they end up in the next chunk with no context.
"""


def chunk_text(text, size=800, overlap=120):
    """
    Walk through text in windows of `size` characters.
    Each window starts `overlap` characters BEFORE the previous
    one ended, so consecutive chunks share a bit of content.

    If you increase overlap, retrieval gets slightly better
    (less risk of losing context) but you store more data.
    10-15% of size is a good ratio.
    """
    if not text:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than size, genius")

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        piece = text[start:end]
        if piece.strip():          # don't store empty whitespace chunks
            chunks.append(piece.strip())
        # Advance, but step BACK by `overlap` so the next chunk
        # picks up where the last one left off (plus a little overlap)
        start = end - overlap
    return chunks