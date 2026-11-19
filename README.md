# DoCrops AI Intelligence Platform 🌾🤖

Production-grade, professional Context-Aware Retrieval-Augmented Generation (RAG) system specialized in Plant Pathology and Crop Disease Diagnosis. Powered by **FastAPI**, **FAISS Vector Database**, and a **small local LLM (Qwen2.5-1.5B via llama.cpp) running fully inside the Space - no external API**.

**Live demo:** https://huggingface.co/spaces/MahmoudGomaa8545/docrops-rag-api

---

## 🚀 Features

* **Conversational RAG Core:** Contextualizes and rewrites follow-up user queries based on chat history to guarantee ultra-precise vector database lookups.
* **Semantic Text Search:** Utilizes `intfloat/multilingual-e5-base` embeddings to match user queries with textbook-level botanical data stored in a dense FAISS index.
* **Pure Academic Outputs:** Designed to return clean, high-quality, professional scientific answers (supporting both Arabic and English) without raw text citation fluff or metadata clutter.
* **Minimalist Responsive UI:** Features a dynamic, modern, single-page chat view that intelligently auto-adapts text alignment (`RTL` / `LTR`) based on the input language.
* **Zero External Dependencies at Runtime:** The LLM runs locally inside the deployment, so the demo works with no API keys and no paid services.

---

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI (Uvicorn ASGI Server)
* **Vector Indexing:** FAISS (Facebook AI Similarity Search)
* **Embedding Model:** `intfloat/multilingual-e5-base`
* **LLM Engine:** Local Qwen2.5-1.5B-Instruct (GGUF) via llama.cpp - no external API
* **Frontend:** HTML5, CSS3 (Dynamic Bi-directional Layout), JavaScript (Marked.js for Markdown parsing)

---

## 📦 Installation & Local Setup

Since this repository tracks large vector index binaries via **Git LFS**, please ensure it is installed before cloning.

### 1. Clone the Repository
```bash
# Initialize Large File Storage
git lfs install

# Clone the repository directly
git clone https://github.com/MahmoudGomaa8545/DoCrops-RAG.git
cd DoCrops-RAG
```

### 2. Install Dependencies
Make sure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Server Locally
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
On first run the app downloads the local LLM (~1.1 GB) automatically, then open your browser and navigate to `http://127.0.0.1:8000` to interact with the system.

---

## 📂 Project Structure
```plaintext
├── app.py                  # Main FastAPI application & Embedded UI Context
├── faiss_index_docrops.bin # Serialized dense FAISS Vector Index
├── metadata_docrops.pkl    # Serialized text metadata chunks mapped to FAISS IDs
├── requirements.txt        # System application dependencies
└── README.md               # Project documentation
```

---

## 🔌 API Specification

### Endpoint: `POST /ask`
Processes user queries, looks up matching scientific contexts, and generates formal pathological evaluations.

#### Request Headers
```http
Content-Type: application/json
```

#### Request Body Structure
```json
{
  "question": "What are the chemical control methods for it?",
  "history": [
    {
      "role": "user",
      "text": "Tell me about Apple Fire Blight."
    },
    {
      "role": "bot",
      "text": "Fire Blight is a destructive bacterial disease caused by Erwinia amylovora..."
    }
  ]
}
```
*Note: For the initial session request, pass `"history": []`.*

#### Response Body Structure
```json
{
  "answer": "Chemical control of Fire Blight involves timely application of copper compounds during early bloom stages and specific antibiotic sprays like streptomycin according to local agricultural guidelines to suppress bacterial multiplication."
}
```

---

## 🌐 Deployment Note
This system is optimized to run inside Hugging Face Spaces Docker or Python environments, with the LLM and embeddings running locally - no secrets or tokens required.
