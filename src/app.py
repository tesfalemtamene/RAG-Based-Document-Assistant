import streamlit as st
import tempfile
import os

from pdf_loader import load_pdf
from text_chunker import chunk_text, clean_chunks
from vector_store import create_vector_store, similarity_search
from llm_model import generate_answer

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Research Assistant (RAG)",
    layout="wide"
)

st.title("📚 AI Research Assistant (RAG System)")
st.markdown("Upload a research paper and ask questions intelligently.")

# -----------------------------
# Session State
# -----------------------------
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing document..."):

        # Save to temp file then clean up afterwards
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        try:
            # Pipeline
            text = load_pdf(temp_path)
            chunks = chunk_text(text)
            chunks = clean_chunks(chunks)
            index, embeddings = create_vector_store(chunks)

            # Store in session
            st.session_state.vector_index = index
            st.session_state.chunks = chunks

        finally:
            # Always clean up the temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    st.success("✅ Document processed successfully!")

    # Show document stats so the user can verify parsing quality
    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Total Characters", f"{len(text):,}")
    col2.metric("🧩 Text Chunks", len(chunks))
    col3.metric("📐 Avg Chunk Size", f"{int(len(text)/max(len(chunks),1)):,} chars")

# -----------------------------
# Question Section
# -----------------------------
query = st.text_input("Ask a question about the document:")

if query:
    if st.session_state.vector_index is None:
        st.warning("⚠️ Please upload a PDF document first before asking a question.")
    else:
        with st.spinner("Searching and generating answer..."):

            # Retrieve top-5 relevant chunks
            results = similarity_search(
                query,
                st.session_state.vector_index,
                st.session_state.chunks,
                top_k=5
            )

            # Combine context from top chunks only
            context = "\n\n".join(chunk for chunk, _ in results)

            # Generate answer
            answer = generate_answer(query, context)

        # -----------------------------
        # Display Answer
        # -----------------------------
        st.subheader("📌 Answer")
        st.write(answer)

        # -----------------------------
        # Show Sources with similarity score
        # -----------------------------
        st.subheader("📖 Source References")

        for i, (chunk, dist) in enumerate(results):
            # Convert L2 distance to a 0-1 similarity score (higher = more similar)
            similarity = round(1 / (1 + dist), 4)
            with st.expander(f"Source {i+1}  —  Similarity: {similarity:.2%}"):
                st.write(chunk)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built with RAG • FAISS • Transformers • Streamlit")