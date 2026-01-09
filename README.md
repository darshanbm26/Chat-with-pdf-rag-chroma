# 📄 Chat with Your PDF (Streamlit + RAG)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.x-FF9A00?style=for-the-badge&logo=huggingface&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2.x-2A5FE7?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/Chroma-Vector_DB-3DDC84?style=for-the-badge)
![SentenceTransformers](https://img.shields.io/badge/Sentence_Transformers-all--MiniLM--L6--v2-4D8FAC?style=for-the-badge)

Lightweight RAG app to **ingest PDFs**, build embeddings with **SentenceTransformers**, store them in **ChromaDB**, and answer questions via **LaMini-T5-738M** served through Streamlit. Includes a simple chat UI variant.

---

## 🚀 Features

- PDF upload and inline preview (Streamlit)
- Text extraction with `PDFMinerLoader`
- Chunking via `RecursiveCharacterTextSplitter`
- Embeddings: `all-MiniLM-L6-v2` (sentence-transformers)
- Vector store: Chroma persistent DB (`db/`)
- Generator: `LaMini-T5-738M` (transformers pipeline)
- Two UIs: `app.py` (Q&A) and `chatbot_app.py` (chat-style)

---

## 📸 Screenshots

![Home Page](<Screenshot (363).png>)
![PDF Upload](<Screenshot (364).png>)
![PDF Details And Preview](<Screenshot (366).png>)
![Chat With PDF](<Screenshot (368).png>)

## 📂 Project Structure (key files)

- `app.py` – main Q&A Streamlit app
- `chatbot.py` – chat view using Streamlit chat UI
- `chatbot_app.py` – PDF upload + chat flow
- `ingest.py` – optional ingestion script (if you prefer CLI ingestion)
- `constants.py` – Chroma settings (legacy kept minimal)
- `requirements.txt` – Python dependencies
- `db/` – Chroma persistence (ignored in git)
- `LaMini-T5-738M/` – local model cache (ignored in git)

---

## 🛠️ Quickstart

### 1) Setup environment

```bash
python -m venv ragenv
./ragenv/Scripts/activate  # Windows
# source ragenv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2) (Optional) Download model locally

By default Hugging Face will cache `LaMini-T5-738M` under your HF cache. If you keep it inside the repo folder, it is already ignored: `LaMini-T5-738M/`.

### 3) Ingest PDFs

Place PDFs in `docs/` and run:

```bash
python ingest.py
```

This builds embeddings into `db/`.

### 4) Run the apps

```bash
streamlit run app.py          # Q&A UI
streamlit run chatbot_app.py  # Upload + chat UI
```

---

## 🧭 Usage Notes

- Upload PDFs via the UI or drop them into `docs/`, then ingest to refresh vectors.
- Chroma DB lives in `db/` (ignored in git). Delete it if you need a clean rebuild.
- If GPU is unavailable, the model loads on CPU with `low_cpu_mem_usage=True`.

---

## 🧹 What not to commit

Already in `.gitignore`:

- `db/`, `offload/` (vector store, offloaded weights)
- `docs/*.pdf`, `docs/**/*.pdf` (your uploaded docs)
- `LaMini-T5-738M/` (local model cache)
- `ragenv/` (local virtualenv)
- Bytecode caches (`__pycache__/`, `*.pyc`)

---

## ✅ Health checks

- `pip check` to verify deps
- `streamlit run app.py` to confirm UI loads
- Ask a question whose answer is in your uploaded PDF to validate retrieval

---

## 🤝 License

MIT License. See `LICENSE` for details.
