
# app.pye
import os
import time
import datetime
import shutil
import pickle
from io import BytesIO

import streamlit as st
import html as html_module
import re
import numpy as np

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import hnswlib

try:
    from langchain.schema import Document
    from langchain.prompts import PromptTemplate
    from langchain.chains.question_answering import load_qa_chain
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

from transformers import pipeline
HF_FALLBACK_MODEL = "google/flan-t5-small"

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="ChatPDF", layout="wide", initial_sidebar_state="expanded")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HNSW_DIR = "hnsw_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVE_K = 4

GEMINI_PREFERRED = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

# ---------------------------
# Embeddings & HNSW Classes
# ---------------------------
class LocalEmbeddings:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        vectors = self.model.encode(texts, show_progress_bar=False)
        return [vec.tolist() if hasattr(vec, "tolist") else list(map(float, vec)) for vec in vectors]

    def embed_query(self, text):
        vec = self.model.encode([text], show_progress_bar=False)[0]
        return vec.tolist() if hasattr(vec, "tolist") else list(map(float, vec))

class LocalHNSW:
    INDEX_FILENAME = "hnsw_index.bin"
    META_FILENAME = "hnsw_meta.pkl"

    def __init__(self, dim: int, space: str = "cosine"):
        self.dim = dim
        self.space = space
        self.index = hnswlib.Index(space=space, dim=dim)
        self._is_initialized = False
        self.id2doc = {}

    @classmethod
    def from_texts(cls, texts, embedding: LocalEmbeddings, space: str = "cosine"):
        vectors = embedding.embed_documents(texts)
        arr = np.array(vectors, dtype=np.float32)
        dim = arr.shape[1]
        obj = cls(dim=dim, space=space)
        obj.index.init_index(max_elements=len(texts), ef_construction=200, M=16)
        ids = np.arange(len(texts), dtype=np.int32)
        obj.index.add_items(arr, ids)
        obj.index.set_ef(50)
        obj._is_initialized = True
        for i, t in enumerate(texts):
            obj.id2doc[int(i)] = {"text": t, "metadata": {}}
        return obj

    def save_local(self, folder):
        os.makedirs(folder, exist_ok=True)
        idx_path = os.path.join(folder, self.INDEX_FILENAME)
        meta_path = os.path.join(folder, self.META_FILENAME)
        self.index.save_index(idx_path)
        with open(meta_path, "wb") as f:
            pickle.dump({"dim": self.dim, "space": self.space, "id2doc": self.id2doc}, f)

    @classmethod
    def load_local(cls, folder, embedding=None):
        meta_path = os.path.join(folder, cls.META_FILENAME)
        idx_path = os.path.join(folder, cls.INDEX_FILENAME)
        if not os.path.exists(meta_path) or not os.path.exists(idx_path):
            raise FileNotFoundError("HNSW index files not found")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        obj = cls(dim=meta["dim"], space=meta["space"])
        obj.index.init_index(max_elements=len(meta["id2doc"]), ef_construction=200, M=16)
        obj.index.load_index(idx_path)
        obj.index.set_ef(50)
        obj._is_initialized = True
        obj.id2doc = {int(k): v for k, v in meta["id2doc"].items()}
        return obj

    def similarity_search(self, query, k=4, embedding: LocalEmbeddings = None):
        if not self._is_initialized:
            return []
        if embedding is None:
            raise ValueError("An embedding object with embed_query is required")
        qvec = embedding.embed_query(query)
        qarr = np.array([qvec], dtype=np.float32)
        labels, distances = self.index.knn_query(qarr, k=k)
        labels = labels[0].tolist()
        results = []
        for lid, dist in zip(labels, distances[0].tolist()):
            rec = self.id2doc.get(int(lid))
            if rec:
                if LANGCHAIN_AVAILABLE:
                    results.append(Document(page_content=rec["text"], metadata={"score": float(dist)}))
                else:
                    results.append({"page_content": rec["text"], "metadata": {"score": float(dist)}})
        return results

# ---------------------------
# PDF Processing
# ---------------------------
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.warning(f"⚠️ Couldn't read {getattr(pdf,'name', 'a file')}")
            continue
    return text

def get_text_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ---------------------------
# Prompt Template - Single Unified Format
# ---------------------------
def build_unified_prompt():
    template = """You are a helpful AI assistant that answers questions based on the provided context.

Instructions:
- Provide a clear, conversational answer based ONLY on the context below
- Structure your response naturally with proper paragraphs
- If the answer requires multiple points, use a brief introduction followed by clear points
- If the answer is not in the context, say "I don't have enough information in the provided documents to answer that question."
- Keep the response concise but complete

Context:
{context}

Question: {question}

Answer:"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ---------------------------
# Answer Generation
# ---------------------------
def generate_answer(prompt_template, docs, question, google_api_key=None):
    if not docs:
        return None, None, "No documents retrieved."
    
    context = "\n\n".join([d.page_content if hasattr(d, "page_content") else d["page_content"] for d in docs])
    
    # Try Gemini first
    if LANGCHAIN_AVAILABLE and google_api_key:
        for model_name in GEMINI_PREFERRED:
            try:
                model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
                chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)
                response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
                
                text = None
                if isinstance(response, dict):
                    for key in ("output_text", "text", "answer", "output"):
                        if key in response and response[key]:
                            text = response[key]
                            break
                elif isinstance(response, str):
                    text = response
                
                if text and text.strip():
                    return clean_answer(text.strip()), model_name, None
            except Exception:
                continue
    
    # Fallback to HuggingFace
    try:
        hf = pipeline("text2text-generation", model=HF_FALLBACK_MODEL, device=-1)
        prompt_text = prompt_template.format(context=context, question=question)
        resp = hf(prompt_text, max_length=256, do_sample=False)
        if isinstance(resp, list) and resp:
            out = resp[0].get("generated_text") or resp[0].get("text") or str(resp[0])
            return clean_answer(out.strip()), HF_FALLBACK_MODEL, None
    except Exception as e:
        return None, None, f"Generation failed: {e}"
    
    return None, None, "All models failed to generate a response."


import re

def clean_answer(text):
    """Clean up the model's response to make it presentable."""
    # Remove common artifacts
    text = re.sub(r"content=(['\"])(.+?)\1", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"\b(additional_kwargs|response_metadata|usage_metadata|id)=\{[^}]*\}", "", text, flags=re.DOTALL)

    # Remove bold markdown and trim
    text = text.replace("**", "").strip()

    # --- New: remove leading bullets / list markers at start of lines ---
    # Remove single-char bullets like "* ", "- ", "• "
    text = re.sub(r"^[\*\-\u2022]\s+", "", text, flags=re.MULTILINE)
    # Remove numbered list markers like "1. ", "2) ", etc.
    text = re.sub(r"^\s*\d+[\.\)]\s+", "", text, flags=re.MULTILINE)
    # Optional: collapse repeated blank lines to a single blank line
    text = re.sub(r"\n\s*\n+", "\n\n", text, flags=re.MULTILINE)

    # Remove duplicate consecutive lines while preserving single occurrences
    lines = text.split("\n")
    cleaned_lines = []
    prev_line = None
    for line in lines:
        # normalize whitespace at line ends/starts
        line = line.strip()
        if line != prev_line and line != "":
            cleaned_lines.append(line)
            prev_line = line
        elif line == "" and prev_line != "":  # keep one blank line between paragraphs
            cleaned_lines.append("")
            prev_line = ""

    return "\n".join(cleaned_lines).strip()


# ---------------------------
# Session State
# ---------------------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "hnsw_ready" not in st.session_state:
        st.session_state.hnsw_ready = os.path.isdir(HNSW_DIR)

# ---------------------------
# Modern UI CSS
# ---------------------------
MODERN_CSS = """
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

/* Main Background */
[data-testid="stAppViewContainer"] { background: #f7f7f8; }

/* ========================
   FORCE SIDEBAR VISIBLE
   ======================== */
/* Primary selector used by Streamlit */
[data-testid="stSidebar"],
/* fallback selectors used by some Streamlit versions */
aside[role="complementary"],
/* handle both expanded and collapsed states explicitly */
[data-testid="stSidebar"][aria-expanded="false"],
[data-testid="stSidebar"][aria-expanded="true"] {
    display: block !important;
    visibility: visible !important;
    transform: none !important;
    width: 21rem !important;
    min-width: 21rem !important;
    max-width: 21rem !important;
    position: relative !important;
    left: 0 !important;
    margin-left: 0 !important;
    opacity: 1 !important;
    z-index: 9999 !important;
}

/* Make sure the collapsed control (toggle) is visible and not overlapping */
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    color: #10a37f;
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    padding: 8px;
    z-index: 10000 !important;
}

/* Keep sidebar content readable */
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #353740 !important;
}

/* Sidebar Markdown color fix (previous hex was invalid) */
[data-testid="stSidebar"] .stMarkdown {
    color: #353740 !important;
}

/* Make sure the sidebar buttons show text */
[data-testid="stSidebar"] button { color: white !important; }

/* Mobile: Make sidebar accessible (override responsive hiding) */
@media (max-width: 768px) {
    [data-testid="stSidebar"],
    aside[role="complementary"] {
        display: block !important;
        visibility: visible !important;
        width: 85vw !important;
        max-width: 300px !important;
        position: fixed !important;
        left: 0 !important;
        top: 0 !important;
        height: 100vh !important;
        overflow: auto !important;
        z-index: 10000 !important;
        background: #ffffff !important;
    }
}

/* Hide Streamlit branding reliably */
#MainMenu, .css-1lcbmhc, [data-testid="stHeader"], header { visibility: hidden !important; height: 0 !important; margin: 0 !important; padding: 0 !important; }

/* Chat Container (main UI) */
.chat-container { max-width: 800px; margin: 0 auto; padding: 20px; }

/* Message Bubbles */
.message { display: flex; margin-bottom: 24px; animation: fadeIn 0.3s ease-in; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.message.user { justify-content: flex-end; }
.message-content { max-width: 80%; padding: 12px 16px; border-radius: 18px; line-height: 1.5; word-wrap: break-word; }
.message.user .message-content { background: #10a37f; color: white; border-bottom-right-radius: 4px; }
.message.assistant .message-content { background: #ffffff; color: #353740; border: 1px solid #e5e5e5; border-bottom-left-radius: 4px; }

/* Avatars */
.avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 14px; margin: 0 12px; flex-shrink: 0; }
.message.user .avatar { background: #10a37f; color: white; order: 2; }
.message.assistant .avatar { background: #19c37d; color: white; }

/* Input Area - ensure text is readable (was white on white) */
.stTextArea textarea { border-radius: 12px; border: 1px solid #d9d9e3; padding: 12px 16px; font-size: 15px; resize: none; color: #353740 !important; background: #fff !important; }
.stTextArea textarea:focus { border-color: #10a37f; box-shadow: 0 0 0 1px #10a37f; }

/* Buttons */
.stButton > button { background: #10a37f; color: white; border: none; border-radius: 8px; padding: 10px 24px; font-weight: 500; transition: all 0.2s; }
.stButton > button:hover { background: #0d8a6a; box-shadow: 0 2px 8px rgba(16, 163, 127, 0.3); }

/* File Uploader - keep contrast for text */
[data-testid="stFileUploader"] { border: 2px dashed #d9d9e3; border-radius: 12px; padding: 20px; background: #ffffff !important; color: #353740 !important; }

/* Status badge */
.status-badge { display: inline-block; padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: 600; margin: 8px 0; text-align: center; }
.status-ready { background: #d1f4e0; color: #0d8a6a; border: 2px solid #10a37f; }
.status-not-ready { background: #fee; color: #d63; border: 2px solid #faa; }

/* small responsive safety */
@media (max-width: 500px) {
  [data-testid="stAppViewContainer"] { padding-left: 0 !important; }
}
</style>
"""



# ---------------------------
# Main App
# ---------------------------
def main():
    st.markdown(MODERN_CSS, unsafe_allow_html=True)
    init_session_state()
    
    # Sidebar for PDF Upload
    with st.sidebar:
        st.markdown("## 📄 Documents")
        st.markdown("")
        
        # Status indicator
        if st.session_state.hnsw_ready:
            st.markdown('<div class="status-badge status-ready">✓ Documents Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-not-ready">⚠ Upload PDFs to Start</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # File upload section
        st.markdown("#### 📤 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload one or more PDF files to chat with"
        )
        
        # Show uploaded file names
        if uploaded_files:
            st.markdown("**Uploaded files:**")
            for file in uploaded_files:
                st.text(f"📄 {file.name}")
            st.markdown("")
        
        if uploaded_files:
            if st.button("🔄 Process Documents", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Extract text
                    status_text.text("📖 Reading PDF files...")
                    progress_bar.progress(20)
                    raw_text = get_pdf_text(uploaded_files)
                    
                    if not raw_text.strip():
                        st.error("❌ No readable text found in PDFs")
                    else:
                        # Create chunks
                        status_text.text("✂️ Splitting text into chunks...")
                        progress_bar.progress(40)
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Build index
                        status_text.text("🔍 Building search index...")
                        progress_bar.progress(60)
                        
                        embeddings = LocalEmbeddings()
                        index = LocalHNSW.from_texts(text_chunks, embedding=embeddings, space='cosine')
                        
                        status_text.text("💾 Saving index...")
                        progress_bar.progress(80)
                        index.save_local(HNSW_DIR)
                        
                        progress_bar.progress(100)
                        st.session_state.hnsw_ready = True
                        
                        status_text.empty()
                        progress_bar.empty()
                        st.success(f"✅ Successfully processed {len(text_chunks)} text chunks from {len(uploaded_files)} file(s)!")
                        time.sleep(1)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
        
        st.markdown("---")
        
        # Chat info
        if st.session_state.messages:
            st.markdown(f"**💬 Messages:** {len(st.session_state.messages)}")
        
        st.markdown("---")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Delete index
        if st.session_state.hnsw_ready:
            if st.button("❌ Delete Documents", use_container_width=True):
                try:
                    if os.path.isdir(HNSW_DIR):
                        shutil.rmtree(HNSW_DIR)
                    st.session_state.hnsw_ready = False
                    st.success("Documents deleted")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Main Chat Interface
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("ChatPDF")
        st.markdown('<p class="subtitle">Ask questions about your documents</p>', unsafe_allow_html=True)
    
    # Display messages or welcome screen
    if not st.session_state.messages:
        if st.session_state.hnsw_ready:
            st.markdown("""
            <div class='welcome-container'>
                <div class='welcome-title'>✅ Ready to Chat!</div>
                <div class='welcome-text'>
                    Your documents are processed and ready.<br>
                    Ask me anything about your PDFs below!
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='welcome-container'>
                <div class='welcome-title'>👋 Welcome to ChatPDF</div>
                <div class='welcome-text'>
                    <strong>Get started:</strong><br>
                    1. Use the sidebar (←) to upload your PDF documents<br>
                    2. Click "Process Documents" to index them<br>
                    3. Ask me anything about your documents!
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Display chat history
        for idx, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                st.markdown(f"""
                <div class='message user'>
                    <div class='message-content'>{html_module.escape(content)}</div>
                    <div class='avatar'>You</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # For assistant, show content with proper formatting
                formatted_content = html_module.escape(content).replace("\n", "<br>")
                st.markdown(f"""
                <div class='message assistant'>
                    <div class='avatar'>AI</div>
                    <div class='message-content'>{formatted_content}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Input area at the bottom
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create input form
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_area(
                "Message",
                placeholder="Ask a question about your documents... (Ctrl+Enter to send)" if st.session_state.hnsw_ready else "Upload and process PDFs first...",
                height=100,
                label_visibility="collapsed",
                key="user_input",
                disabled=not st.session_state.hnsw_ready
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit_disabled = not st.session_state.hnsw_ready
            submit = st.form_submit_button(
                "Send" if st.session_state.hnsw_ready else "📄 Upload PDFs First",
                use_container_width=True,
                disabled=submit_disabled
            )
        
        if submit and user_input and user_input.strip():
            if not st.session_state.hnsw_ready:
                st.error("⚠️ Please upload and process PDF documents first!")
            else:
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input.strip()
                })
                
                # Show thinking indicator
                with st.spinner("🤔 Searching documents and generating answer..."):
                    # Retrieve relevant documents
                    try:
                        embeddings = LocalEmbeddings()
                        db = LocalHNSW.load_local(HNSW_DIR, embedding=embeddings)
                        docs = db.similarity_search(user_input, k=RETRIEVE_K, embedding=embeddings)
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error loading documents: {str(e)}"
                        })
                        st.rerun()
                    
                    if docs:
                        prompt_template = build_unified_prompt()
                        google_api_key = st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
                        answer, model_used, error = generate_answer(
                            prompt_template,
                            docs,
                            user_input,
                            google_api_key=google_api_key
                        )
                        
                        if answer:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer
                            })
                        else:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"I couldn't generate an answer. Error: {error if error else 'Unknown error'}"
                            })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "I couldn't find any relevant information in the documents to answer your question."
                        })
                
                # Rerun to display new messages
                st.rerun()

if __name__ == "__main__":
    main()












