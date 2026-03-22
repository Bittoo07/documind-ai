import streamlit as st
import os
import tempfile
from rag_engine import (
    load_document,
    split_documents,
    create_vector_store,
    build_qa_chain
)

# ── PAGE SETUP ────────────────────────────────────────────────────────────
# This sets the browser tab title and page layout
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 DocuMind AI")
st.caption("Upload any document and ask questions about it using AI")


# ── SIDEBAR ───────────────────────────────────────────────────────────────
# Everything inside "with st.sidebar" appears on the left panel
with st.sidebar:
    st.header("📄 Upload Your Document")

    # File uploader — accepts PDF and TXT only
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"]
    )

    # This button triggers the whole ingestion pipeline
    if st.button("⚡ Process Document", use_container_width=True):

        if uploaded_file is None:
            # If no file selected, show a warning
            st.warning("Please upload a file first!")

        else:
            # Show a spinner while processing
            with st.spinner("Reading and indexing document..."):

                # Save uploaded file to a temporary location on disk
                # We do this because PyPDFLoader needs a real file path
                suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(uploaded_file.read())
                    temp_path = f.name
                # temp_path is now something like: C:\Users\...\tmpXXXX.pdf

                # Run the full RAG pipeline
                docs         = load_document(temp_path)
                chunks       = split_documents(docs)
                vector_store = create_vector_store(chunks)
                chain        = build_qa_chain(vector_store)

                # Save chain and chat history in session_state
                # session_state keeps data alive between interactions
                st.session_state.chain         = chain
                st.session_state.chat_history  = []
                st.session_state.doc_name      = uploaded_file.name

                # Clean up the temp file
                os.unlink(temp_path)

            st.success(f"✅ Ready! Indexed {len(chunks)} chunks.")

    # Show which document is loaded
    if "doc_name" in st.session_state:
        st.info(f"📄 Loaded: {st.session_state.doc_name}")

        # Reset button clears everything
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            del st.session_state.chain
            del st.session_state.chat_history
            del st.session_state.doc_name
            st.rerun()


# ── MAIN CHAT AREA ────────────────────────────────────────────────────────

if "chain" not in st.session_state:
    # No document loaded yet — show instructions
    st.info("👈 Upload a PDF or TXT file from the sidebar to get started")

else:
    # Show all previous messages in the chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input box at the bottom of the page
    question = st.chat_input("Ask anything about your document...")

    if question:
        # Show the user's question in the chat
        with st.chat_message("user"):
            st.write(question)

        # Save user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        # Get answer from the RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                # This is where the magic happens!
                # chain.invoke sends the question through:
                # retriever → prompt → GPT → answer
                answer = st.session_state.chain.invoke(question)

            st.write(answer)

        # Save assistant answer to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })