import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Simple RAG Bot", layout="wide")
st.title("📄 Instant RAG Chatbot")

# --- Sidebar for Data Ingestion ---
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    process_btn = st.button("Process Document")

# --- Backend Logic ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file and process_btn:
    with st.spinner("Processing..."):
        # Save temp file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and Split
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # Create Vector Store
        st.session_state.vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=OpenAIEmbeddings()
        )
        st.success("Document indexed!")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about your document"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vectorstore:
        # Retrieval Chain
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm, 
            retriever=st.session_state.vectorstore.as_retriever()
        )
        
        with st.chat_message("assistant"):
            response = qa_chain.invoke(prompt)["result"]
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload and process a document first!")
