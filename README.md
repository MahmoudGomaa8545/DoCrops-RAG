# DoCrops AI Intelligence Platform 🌾🤖

Production-grade, professional Context-Aware Retrieval-Augmented Generation (RAG) system specialized in Plant Pathology and Crop Disease Diagnosis. Powered by **FastAPI**, **FAISS Vector Database**, and **Hugging Face Serverless Inference API**.

---

## 🚀 Features

* **Conversational RAG Core:** Contextualizes and rewrites follow-up user queries based on chat history to guarantee ultra-precise vector database lookups.
* **Semantic Text Search:** Utilizes `intfloat/multilingual-e5-base` embeddings to match user queries with textbook-level botanical data stored in a dense FAISS index.
* **Pure Academic Outputs:** Designed to return clean, high-quality, professional scientific answers (supporting both Arabic and English) without raw text citation fluff or metadata clutter.
* **Minimalist Responsive UI:** Features a dynamic, modern, single-page chat view that intelligently auto-adapts text alignment (`RTL` / `LTR`) based on the input language.

---

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI (Uvicorn ASGI Server)
* **Vector Indexing:** FAISS (Facebook AI Similarity Search)
* **Embedding Model:** `intfloat/multilingual-e5-base`
* **LLM Engine:** Serverless Inference Client via Hugging Face (`Qwen/Qwen2.5-7B-Instruct`)
* **Frontend:** HTML5, CSS3 (Dynamic Bi-directional Layout), JavaScript (Marked.js for Markdown parsing)

---

## 📦 Installation & Local Setup

Since this repository tracks large vector index binaries via **Git LFS**, please ensure it is installed before cloning.

### 1. Clone the Repository
```bash
# Initialize Large File Storage
git lfs install

# Clone the repository
git clone [https://huggingface.co/spaces/MahmoudGomaa8545/docrops-rag-api](https://huggingface.co/spaces/MahmoudGomaa8545/docrops-rag-api)
cd docrops-rag-api
