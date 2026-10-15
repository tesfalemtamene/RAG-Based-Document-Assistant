# 📚 RAG-Based Document Assistant

An AI-powered research assistant that lets you upload any PDF document and ask natural language questions about it. Built using **Retrieval-Augmented Generation (RAG)** — combining semantic search with a local LLM to produce grounded, context-aware answers.

---

## 🧠 How It Works

```
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Index
                                                              ↓
User Question → Query Embedding → Similarity Search → Context Retrieval
                                                              ↓
                                              flan-t5-base LLM → Answer
```

1. **PDF Loader** — Extracts raw text from the uploaded PDF
2. **Text Chunker** — Splits text into overlapping chunks (800 chars, 150 overlap)
3. **Vector Store** — Generates sentence embeddings (`all-mpnet-base-v2`) and indexes them in FAISS
4. **LLM** — `google/flan-t5-base` (seq2seq) answers the question using only the retrieved context

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| PDF parsing | `pypdf` |
| Text splitting | `langchain-community` |
| Embeddings | `sentence-transformers` (`all-mpnet-base-v2`) |
| Vector search | `faiss-cpu` |
| LLM | `transformers` (`google/flan-t5-base`) |
| UI | `streamlit` |

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/tesfalemtamene/RAG-Based-Document-Assistant.git
cd RAG-Based-Document-Assistant
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install faiss-cpu
```

> **Note:** `flan-t5-base` (~1 GB) and `all-mpnet-base-v2` (~420 MB) will be downloaded automatically from Hugging Face on first run.

---

## 🚀 Running the App

### Option A — Streamlit Web UI (recommended)
```bash
cd src
streamlit run app.py
```
Open **http://localhost:8501** in your browser, upload a PDF, and start asking questions.

### Option B — Command-line pipeline
Place a PDF at `data/sample_paper.pdf`, then:
```bash
cd src
python main.py
```

---

## 📁 Project Structure

```
├── data/                  # Place your PDF files here (gitignored)
├── src/
│   ├── app.py             # Streamlit web UI
│   ├── main.py            # CLI pipeline entry point
│   ├── pdf_loader.py      # PDF text extraction
│   ├── text_chunker.py    # Text splitting and cleaning
│   ├── vector_store.py    # FAISS index creation and similarity search
│   └── llm_model.py       # flan-t5 answer generation
├── requirements.txt
└── .gitignore
```

---

## 💡 Features

- 📄 Upload any PDF research paper or document
- 🔍 Semantic similarity search (not keyword matching)
- 🤖 Local LLM — no API keys or internet required after first download
- 📊 Document stats shown after upload (characters, chunks, avg size)
- 📖 Source references with similarity scores for every answer
- ⚠️ Graceful handling when no document is uploaded

---

## 📌 Notes

- The model answers based **only** on the uploaded document — it won't hallucinate from general knowledge
- For best results, use clearly written academic papers or reports
- GPU is not required; the model runs on CPU
