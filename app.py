import streamlit as st
import os
from io import BytesIO
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# Attempt to import FAISS; if unavailable, fall back to hnswlib
try:
    import faiss
    use_faiss = True
except ImportError:
    use_faiss = False
    import hnswlib

# For generative model
USE_GOOGLE = False
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None) or os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    from langchain_chat_google_genai import ChatGoogleGenerativeAI
    # Initialize Google Gemini model via LangChain
    chat_model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-1.5-pro", temperature=0.7)
    USE_GOOGLE = True
else:
    # Fallback to HuggingFace pipeline (e.g. Flan-T5 base)
    from transformers import pipeline
    chat_model = pipeline("text2text-generation", model="google/flan-t5-base")

# Initialize embedding model (SentenceTransformer)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# UI layout
st.set_page_config(page_title="PDF Chat with Gemini 2.0", page_icon="📄")
st.title("Chat with PDFs (RAG App)")

# File upload
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
process = st.button("Process PDFs and Build Index")

# Global variables for index and text chunks
if 'vector_index' not in st.session_state:
    st.session_state['vector_index'] = None
    st.session_state['chunks'] = None

if process:
    all_text = ""
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            all_text += page.extract_text() + "\n"
    # Simple chunking (split by fixed character length)
    chunk_size = 1000
    chunks = [all_text[i:i+chunk_size] for i in range(0, len(all_text), chunk_size)]
    st.session_state['chunks'] = chunks
    
    # Embed chunks
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    
    if use_faiss:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
    else:
        # HNSWLib index
        dim = embeddings.shape[1]
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=len(chunks), ef_construction=200, M=16)
        index.add_items(embeddings)
        index.set_ef(50)
    
    st.session_state['vector_index'] = index
    st.success(f"Indexed {len(chunks)} text chunks from PDFs.")
    
    # Clear any previous Q&A
    st.session_state['history'] = []

# Chat interface
question = st.text_input("Enter your question about the PDF content:")
if question:
    if st.session_state['vector_index'] is None:
        st.warning("Please upload PDFs and click Process first.")
    else:
        # Retrieve relevant chunks
        q_embed = embedding_model.encode([question], convert_to_numpy=True)
        k = 3  # retrieve top 3
        if use_faiss:
            D, I = st.session_state['vector_index'].search(q_embed, k)
        else:
            labels, distances = st.session_state['vector_index'].knn_query(q_embed, k=k)
            I = labels
        
        retrieved_chunks = [st.session_state['chunks'][i] for i in I[0]]
        context = "\n\n".join(retrieved_chunks)
        
        # Prepare prompt for the LLM
        if USE_GOOGLE:
            prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"
            answer = chat_model.invoke(prompt)
        else:
            prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
            response = chat_model(prompt, max_length=200, do_sample=False)
            answer = response[0]['generated_text']
        
        # Display the conversation
        st.session_state.setdefault('history', []).append((question, answer))
        for i, (q, a) in enumerate(st.session_state['history']):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")
