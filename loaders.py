# loaders.py
"""
The bouncer at the door. Give it a path or a URL, it hands back
plain text. The rest of the system only ever sees plain text,
so if we want to support a new file type later (like .pptx),
this is the ONLY file we need to touch.
"""

import os
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document
import openpyxl


def load_pdf(path):
    """
    PDFs don't store "paragraphs" — they store individual glyphs
    at (x, y) positions. pypdf does a best-effort job reconstructing
    the text. It's not perfect but it's good enough for 95% of PDFs.

    Note: this FAILS on scanned PDFs (images of text). Those need OCR
    — add pytesseract later if you hit that case.
    """
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def load_docx(path):
    """
    Word docs are cleaner — python-docx gives us real paragraphs.
    But tables are separate objects, so we walk those too and flatten
    them into pipe-separated rows. That way the LLM sees
    "Name | Role | Salary" style lines instead of a wall of disconnected words.
    """
    doc = Document(path)
    lines = []
    for p in doc.paragraphs:
        if p.text.strip():
            lines.append(p.text)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def load_xlsx(path):
    """
    Spreadsheets are the trickiest. A bare row of numbers means nothing
    without column headers. So we grab the first row as headers, then
    format every subsequent row as "col1=val, col2=val".

    Result looks like:
      ### Sheet: Q3
      Region=EMEA, Revenue=50000, Growth=12%
      Region=APAC, Revenue=80000, Growth=9%

    Now the LLM can actually understand what it's looking at.
    """
    wb = openpyxl.load_workbook(path, data_only=True)  # data_only=True reads values, not formulas
    out = []
    for sheet in wb.worksheets:
        out.append(f"### Sheet: {sheet.title}")
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            continue
        # First row = headers. Handle blank header cells with col0, col1, ...
        headers = [str(h) if h is not None else f"col{i}"
                   for i, h in enumerate(rows[0])]
        for row in rows[1:]:
            pairs = [f"{headers[i]}={v}"
                     for i, v in enumerate(row) if v is not None]
            if pairs:
                out.append(", ".join(pairs))
    return "\n".join(out)


def load_txt(path):
    """
    Straightforward, but some old files are latin-1 not utf-8.
    Try utf-8 first, fall back gracefully instead of crashing.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def load_url(url):
    """
    Grab a webpage and strip out the noise. A raw HTML page is
    ~80% nav bars, cookie banners, and ads. BeautifulSoup lets us
    delete those tags outright, then extract just the visible text.
    """
    headers = {"User-Agent": "Mozilla/5.0"}  # some sites block bare requests
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # Nuke the stuff we don't want — nav, footers, scripts, etc.
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    # get_text gives us everything between tags. Then collapse
    # all the blank lines HTML leaves behind.
    raw = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return "\n".join(lines)


def load_any(source):
    """
    The dispatcher. URL or file? File extension? Just picks
    the right loader above and calls it.

    Adding a new format later = add one elif here + one loader function.
    """
    if source.startswith(("http://", "https://")):
        return load_url(source)
    ext = os.path.splitext(source)[1].lower()
    if ext == ".pdf":
        return load_pdf(source)
    if ext == ".docx":
        return load_docx(source)
    if ext in (".xlsx", ".xls"):
        return load_xlsx(source)
    if ext in (".txt", ".md"):
        return load_txt(source)
    raise ValueError(f"Don't have a loader for {ext} yet — add one to loaders.py")