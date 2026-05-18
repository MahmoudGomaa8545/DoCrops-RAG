import os
import pickle
import faiss
import numpy as np
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Enforce fast model downloading using the mirror endpoint
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

app = FastAPI(title="DoCrops Production RAG API", description="Stable Serverless Inference Tier")

# 1. Initialize and configure the Hugging Face Inference Client
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

if HF_TOKEN:
    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
    print(f"Success: Inference Client initialized securely for {MODEL_ID}.", flush=True)
else:
    client = None
    print("CRITICAL WARNING: HF_TOKEN environment variable is missing from Secrets.", flush=True)

# 2. Load RAG resources at server startup
print("Loading FAISS Index and Multilingual Embedding Model...", flush=True)
VECTOR_DB_PATH = "faiss_index_docrops.bin"
METADATA_PATH  = "metadata_docrops.pkl"

if not os.path.exists(VECTOR_DB_PATH) or not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("FAISS Index or Metadata files were not found in the root directory.")

index = faiss.read_index(VECTOR_DB_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

embedding_model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
print("All resources loaded successfully. AI Core is live.", flush=True)

# 3. Data structures for Chat History
class ChatMessage(BaseModel):
    role: str
    text: str

class QueryRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []

def search_faiss(query, k=3):
    formatted_query = f"query: {query}"
    query_vector = embedding_model.encode(formatted_query, convert_to_numpy=True, normalize_embeddings=True)
    query_vector = np.array([query_vector]).astype(np.float32)
    distances, indices = index.search(query_vector, k)
    return [metadata[idx] for idx in indices[0] if idx != -1]

# English Gemini-Inspired Interface
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en" dir="ltr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DoCrops AI Platform</title>
        <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
                background-color: #f4f6f8; color: #1f1f1f; display: flex; justify-content: center; align-items: center; height: 100vh;
            }
            .chat-container { 
                background: #ffffff; width: 100%; max-width: 700px; height: 85vh; display: flex; flex-direction: column; 
                border-radius: 16px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); border: 1px solid #e3e3e3; overflow: hidden;
            }
            .chat-header { padding: 20px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; }
            .title { font-size: 16px; font-weight: 600; color: #1f1f1f; }
            .clear-btn { background: transparent; border: none; color: #757575; font-size: 13px; cursor: pointer; font-weight: 500; }
            .clear-btn:hover { color: #d32f2f; }
            .chat-box { flex: 1; padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 20px; background-color: #ffffff; }
            .message-wrapper { display: flex; width: 100%; flex-direction: column; align-items: flex-start; }
            .sender-name { font-size: 13px; font-weight: 700; color: #5f6368; margin-bottom: 4px; }
            .message-wrapper.user .sender-name { color: #1f1f1f; }
            .message-wrapper.bot .sender-name { color: #34a853; }
            .message { max-width: 100%; line-height: 1.6; font-size: 15px; word-wrap: break-word; }
            .user-message { background-color: #f0f4f9; padding: 10px 16px; border-radius: 16px; font-weight: 500; }
            .bot-message { background-color: transparent; width: 100%; }
            .bot-message p { margin-bottom: 12px; }
            .bot-message ul, .bot-message ol { padding-left: 20px; padding-right: 0; margin-bottom: 12px; }
            .bot-message li { margin-bottom: 4px; }
            .bot-message strong { font-weight: 700; color: #000; }
            .ar-text { direction: rtl !important; text-align: right !important; font-family: 'Cairo', sans-serif !important; }
            .ar-text ul, .ar-text ol { padding-right: 20px; padding-left: 0; }
            .en-text { direction: ltr !important; text-align: left !important; }
            .input-container { padding: 16px 24px 24px 24px; background: #ffffff; border-top: 1px solid #eee; }
            .input-wrapper { display: flex; align-items: center; background: #f0f4f9; border-radius: 24px; padding: 4px 16px; }
            input { flex: 1; font-family: inherit; padding: 10px 0; border: none; background: transparent; outline: none; font-size: 15px; color: #1f1f1f; }
            input::placeholder { color: #757575; }
            .send-btn { background: transparent; border: none; cursor: pointer; color: #1f1f1f; font-weight: 600; font-size: 14px; padding: 0 8px; }
            .send-btn:hover { color: #34a853; }
            .send-btn:disabled { color: #c4c7c5; cursor: not-allowed; }
            .loading-text { color: #757575; font-style: italic; font-size: 14px; }
        </style>
    </head>
    <body>

    <div class="chat-container">
        <div class="chat-header">
            <div class="title">DoCrops AI Assistant</div>
            <button class="clear-btn" onclick="clearChat()">Clear Chat</button>
        </div>
        <div class="chat-box" id="chatBox">
            <div class="message-wrapper bot">
                <div class="sender-name">DoCrops AI</div>
                <div class="message bot-message en-text">Welcome. System is live and fully optimized. Please enter your query regarding plant pathology.</div>
            </div>
        </div>
        <div class="input-container">
            <div class="input-wrapper">
                <input type="text" id="userInput" placeholder="Type your question here..." onkeypress="handleKeyPress(event)">
                <button id="sendBtn" class="send-btn" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>

    <script>
        let conversationHistory = [];

        function handleKeyPress(event) { if (event.key === 'Enter') sendMessage(); }
        function isArabic(text) { return /[\u0600-\u06FF]/ .test(text); }

        function clearChat() {
            conversationHistory = [];
            document.getElementById('chatBox').innerHTML = `
                <div class="message-wrapper bot">
                    <div class="sender-name">DoCrops AI</div>
                    <div class="message bot-message en-text">Chat cleared. How can I help you?</div>
                </div>
            `;
        }

        async function sendMessage() {
            const inputEl = document.getElementById('userInput');
            const buttonEl = document.getElementById('sendBtn');
            const chatBox = document.getElementById('chatBox');
            const query = inputEl.value.trim();

            if (!query) return;

            const userLangClass = isArabic(query) ? 'ar-text' : 'en-text';
            chatBox.innerHTML += `
                <div class="message-wrapper user">
                    <div class="sender-name">You</div>
                    <div class="message user-message ${userLangClass}">${query}</div>
                </div>
            `;
            inputEl.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;

            inputEl.disabled = true;
            buttonEl.disabled = true;

            const loadingId = 'loading-' + Date.now();
            chatBox.innerHTML += `
                <div class="message-wrapper bot" id="${loadingId}">
                    <div class="sender-name">DoCrops AI</div>
                    <div class="message bot-message loading-text">Thinking...</div>
                </div>
            `;
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: query, history: conversationHistory })
                });
                
                const data = await response.json();
                document.getElementById(loadingId).remove();

                if (response.ok) {
                    const botLangClass = isArabic(data.answer) ? 'ar-text' : 'en-text';
                    const parsedHTML = marked.parse(data.answer);
                    
                    chatBox.innerHTML += `
                        <div class="message-wrapper bot">
                            <div class="sender-name">DoCrops AI</div>
                            <div class="message bot-message ${botLangClass}">
                                ${parsedHTML}
                            </div>
                        </div>
                    `;

                    conversationHistory.push({ role: "user", text: query });
                    conversationHistory.push({ role: "bot", text: data.answer });
                } else {
                    const errDetail = data.detail ? data.detail : "Unknown API Error";
                    
                    const msgWrapper = document.createElement('div');
                    msgWrapper.className = 'message-wrapper bot';
                    
                    const senderDiv = document.createElement('div');
                    senderDiv.className = 'sender-name';
                    senderDiv.textContent = 'DoCrops AI';
                    
                    const contentDiv = document.createElement('div');
                    contentDiv.className = 'message bot-message en-text';
                    contentDiv.style.color = '#d32f2f';
                    contentDiv.style.backgroundColor = '#ffebee';
                    contentDiv.style.padding = '10px';
                    contentDiv.style.borderRadius = '8px';
                    contentDiv.style.whiteSpace = 'pre-wrap';
                    contentDiv.textContent = "Backend Error: " + errDetail;
                    
                    msgWrapper.appendChild(senderDiv);
                    msgWrapper.appendChild(contentDiv);
                    chatBox.appendChild(msgWrapper);
                }
            } catch (error) {
                document.getElementById(loadingId).remove();
                chatBox.innerHTML += `
                    <div class="message-wrapper bot">
                        <div class="sender-name">DoCrops AI</div>
                        <div class="message bot-message en-text" style="color:#d32f2f;">Error: Connection to server failed.</div>
                    </div>
                `;
            }

            inputEl.disabled = false;
            buttonEl.disabled = false;
            inputEl.focus();
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>

    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/ask")
def ask_bot(request: QueryRequest):
    question = request.question
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not client:
        raise HTTPException(status_code=500, detail="HF_TOKEN is missing or InferenceClient not initialized.")

    try:
        search_query = question
        history_context = ""
        
        # Context Re-writing
        if request.history:
            try:
                for msg in request.history:
                    speaker = "User" if msg.role == "user" else "Bot"
                    history_context += f"{speaker}: {msg.text}\n"
                    
                rewrite_prompt = f"Given the following conversation history and a follow-up question, rewrite the follow-up question into a standalone, descriptive search query in English that contains the explicit plant disease name.\n\nConversation History:\n{history_context}\n\nFollow-up Question: {question}\nStandalone Search Query:"
                
                rewrite_response = client.chat_completion(
                    messages=[{"role": "user", "content": rewrite_prompt}],
                    max_tokens=80,
                    temperature=0.1
                )
                if rewrite_response and rewrite_response.choices:
                    search_query = rewrite_response.choices[0].message.content.strip()
                else:
                    search_query = question
            except Exception:
                search_query = question

        # Vector DB Lookup
        matched_docs = search_faiss(search_query, k=3)
        context = "".join([f"Page {d['page_number']}:\n{d['text']}\n\n" for d in matched_docs])

        # Main Academic Prompt
        prompt = f"""
You are an advanced, highly formal, and academic Plant Pathology AI Core for the DoCrops project. Answer the user's latest question directly, rigorously, and professionally using the provided technical textbook context and history.

Strict Evaluation Constraints:
1- Language Consistency: Detect the language of the user's question. If the user asks in Arabic, answer in impeccable, formal, and structured academic Arabic. If the user asks in English, reply in highly professional scientific English.
2- Absolute Formatting Restriction: Do NOT mention any book titles, authors, page numbers, or external citations anywhere in your response text. Provide the pure scientific and practical answer directly. Do not use phrases like "According to the text" or "Based on page X".
3- Scientific Accuracy: Deliver high-quality, factual information derived directly from the provided textbook text. Never use colloquial terms, slang, or conversational fluff.
4- Fallback Rule: If the provided textbook context has absolutely no relevance to the question, state formally in the matching language that there is insufficient data available in the current knowledge core.

Conversation History:
{history_context}

Technical Textbook Context (Agrios):
{context}

User's Latest Question:
{question}

Formal Scientific Response:
"""

        # Final Response Generation via Hugging Face Serverless API
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.3
        )
        
        ai_answer = response.choices[0].message.content.strip()
        
        return {
            "answer": ai_answer
        }

    except Exception as e:
        print("\n=== [CRITICAL API EXCEPTION TRACEBACK] ===", flush=True)
        traceback.print_exc()
        print("===========================================\n", flush=True)
        raise HTTPException(status_code=500, detail=f"Server Exception: {str(e)}")
