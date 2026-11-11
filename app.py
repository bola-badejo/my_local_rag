# app.py
"""
Streamlit web UI. Run with:  streamlit run app.py

Left sidebar: upload files or paste URLs, hit Ingest
Main area: chat with your documents

Nothing here talks to the cloud. Uploads go to temp files,
get loaded/chunked/embedded locally, and the temporary file
is discarded after.
"""

import os
import tempfile
import streamlit as st
from config import DOCS_DIR
from ingest import ingest_one, find_all_docs
from qa import answer_question
from vectorstore import get_collection

st.set_page_config(page_title="Venia Business Hub Assistant", page_icon="💼", layout="wide")
st.title("💼 Venia Business Hub Assistant")
st.caption("Ask anything about Venia's services, pricing, locations and offerings.")


@st.cache_resource(show_spinner="Preparing the knowledge base...")
def _bootstrap_index():
    """
    Streamlit Cloud's filesystem is ephemeral — the vector DB is wiped
    whenever the container restarts. If we boot and the collection is
    empty but docs/ has files, ingest them automatically so the demo
    survives cold starts without manual re-uploading.
    """
    col = get_collection()
    if col.count() > 0:
        return col.count()
    sources = find_all_docs()
    for src in sources:
        try:
            ingest_one(src)
        except Exception as e:
            print(f"Boot ingest failed for {src}: {e}")
    return get_collection().count()


_bootstrap_count = _bootstrap_index()

# --- Sidebar: ingestion controls ---
with st.sidebar:
    st.header("📥 Add Documents")

    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "xlsx", "xls", "txt", "md"],
        accept_multiple_files=True,
    )
    url_input = st.text_input("Or paste an article URL")

    if st.button("Ingest", type="primary"):
        sources = []
        # Save uploads to temp files because our loaders want paths
        for f in uploaded or []:
            suffix = os.path.splitext(f.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(f.getbuffer())
            tmp.close()
            sources.append((tmp.name, f.name))

        if url_input.strip():
            sources.append((url_input.strip(), url_input.strip()))

        if not sources:
            st.warning("Upload something or paste a URL first.")
        else:
            with st.spinner("Loading, chunking, embedding (this takes a minute per file)..."):
                for path, label in sources:
                    with st.status(f"Ingesting {label}...", expanded=False) as status:
                        ingest_one(path)
                        status.update(label=f"✅ {label}", state="complete")

            # clean up temp files
            for path, _ in sources:
                if path.startswith(tempfile.gettempdir()):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            st.success(f"Done. Database has {get_collection().count()} chunks.")

    # Show current DB size
    try:
        n = get_collection().count()
        st.metric("Chunks in database", n)
    except Exception:
        pass

    st.divider()
    if st.button("🗑️ Wipe database"):
        from vectorstore import reset
        reset()
        st.success("Wiped. Re-ingest to rebuild.")

# --- Main: chat ---
if "history" not in st.session_state:
    st.session_state.history = []

# Redraw previous messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# New question
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking locally..."):
            result = answer_question(prompt)
        st.write(result["answer"])
        if result["sources"]:
            st.caption(f"📚 Sources: {', '.join(result['sources'])}")

    st.session_state.history.append({"role": "assistant", "content": result["answer"]})