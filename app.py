# app.py
import os
import time
import shutil
import pickle
from io import BytesIO

import streamlit as st
import html as html_module
import re
import numpy as np

# Try optional heavy imports — handle gracefully if missing
try:
    from PyPDF2 import PdfReader
    HAVE_PYPDF2 = True
except Exception:
    HAVE_PYPDF2 = False

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except Exception:
    HAVE_SENTENCE_TRANSFORMERS = False

try:
    import hnswlib
    HAVE_HNSWLIB = True
except Exception:
    HAVE_HNSWLIB = False

# LangChain & Google generative attempt
try:
    from langchain.schema import Document
    from langchain.prompts import PromptTemplate
    from langchain.chains.question_answering import load_qa_chain
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# Transformers fallback
try:
    from transformers import pipeline
    HF_FALLBACK_MODEL = "google/flan-t5-small"
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False
    HF_FALLBACK_MODEL = None

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
# Local embeddings & HNSW (optional)
# ---------------------------
class LocalEmbeddings:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        if not HAVE_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers not installed")
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
        if not HAVE_HNSWLIB:
            raise RuntimeError("hnswlib not installed")
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
            if HAVE_PYPDF2:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            else:
                # fallback placeholder if PyPDF2 not installed
                text += f"[Could not extract {getattr(pdf,'name','file')} — install PyPDF2 to enable extraction]\n\n"
        except Exception:
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
    return PromptTemplate(template=template, input_variables=["context", "question"]) if LANGCHAIN_AVAILABLE else None

# ---------------------------
# Answer Generation (LangChain/Gemini fallback/HF)
# ---------------------------
def generate_answer(prompt_template, docs, question, google_api_key=None):
    """
    Tries LangChain + Gemini (if available/keys present), else HF pipeline fallback.
    `docs` should be a list of objects with attribute page_content OR dicts with 'page_content'.
    Returns (answer_text or None, model_used_or_none, error_or_none)
    """
    if not docs:
        return None, None, "No documents retrieved."

    context = "\n\n".join([d.page_content if hasattr(d, "page_content") else d["page_content"] for d in docs])

    # Try Gemini via LangChain if available
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

    # Fallback to transformers pipeline if available
    if HAVE_TRANSFORMERS:
        try:
            hf = pipeline("text2text-generation", model=HF_FALLBACK_MODEL, device=-1)
            prompt_text = prompt_template.format(context=context, question=question) if prompt_template else f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            resp = hf(prompt_text, max_length=256, do_sample=False)
            if isinstance(resp, list) and resp:
                out = resp[0].get("generated_text") or resp[0].get("text") or str(resp[0])
                return clean_answer(out.strip()), HF_FALLBACK_MODEL, None
        except Exception as e:
            return None, None, f"Generation failed: {e}"

    # No model available
    return None, None, "No generation model available (no Gemini/HF)."

# ---------------------------
# Clean answer utility
# ---------------------------
def clean_answer(text):
    """Clean up the model's response to make it presentable."""
    # Remove common artifacts
    text = re.sub(r"content=(['\"])(.+?)\1", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"\b(additional_kwargs|response_metadata|usage_metadata|id)=\{[^}]*\}", "", text, flags=re.DOTALL)

    # Remove bold markdown and trim
    text = text.replace("**", "").strip()

    # Remove leading bullets / list markers at start of lines
    text = re.sub(r"^[\*\-\u2022]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[\.\)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\s*\n+", "\n\n", text, flags=re.MULTILINE)

    # Remove duplicate consecutive lines while preserving one blank line
    lines = text.split("\n")
    cleaned_lines = []
    prev_line = None
    for line in lines:
        line = line.strip()
        if line != prev_line and line != "":
            cleaned_lines.append(line)
            prev_line = line
        elif line == "" and prev_line != "":
            cleaned_lines.append("")
            prev_line = ""

    return "\n".join(cleaned_lines).strip()

# ---------------------------
# Simple lexical retriever (fallback)
# ---------------------------
def lexical_score(query, text):
    q_tokens = set(re.findall(r"\w+", query.lower()))
    if not q_tokens:
        return 0
    t_tokens = set(re.findall(r"\w+", text.lower()))
    overlap = q_tokens & t_tokens
    return len(overlap) / (len(q_tokens) or 1)

def retrieve_top_chunks_lexical(query, chunks, top_k=RETRIEVE_K):
    if not chunks:
        return []
    scored = [(i, lexical_score(query, c), c) for i, c in enumerate(chunks)]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [ {"page_content": c, "metadata": {}} for i, score, c in scored[:top_k] if score > 0 ]
    return top

# ---------------------------
# Session State
# ---------------------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "hnsw_ready" not in st.session_state:
        st.session_state.hnsw_ready = os.path.isdir(HNSW_DIR)
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "use_local_hnsw" not in st.session_state:
        st.session_state.use_local_hnsw = False
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

# ---------------------------
# Modern UI CSS (capsule + animations + theme)
# ---------------------------
MODERN_CSS = """
<style>
/* ---------- Fonts & base ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { box-sizing: border-box; font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; }

/* Theme variables (light + dark) */
:root[data-theme="light"] {
  --bg: #f5f7f9;
  --card: #ffffff;
  --text: #0f1720;
  --muted: #6b7280;
  --accent: #10a37f;
  --accent-strong: #0d8a6a;
  --border: #e6e9ee;
  --uploader-bg: #ffffff;
  --input-bg: #ffffff;
  --input-text: #0f1720;
  --shadow-sm: 0 6px 18px rgba(16, 163, 127, 0.06);
  --shadow-lg: 0 20px 50px rgba(16, 163, 127, 0.06);
}
:root[data-theme="dark"] {
  --bg: #07101a;
  --card: #07101a;
  --text: #e6eef6;
  --muted: #9aa6b2;
  --accent: #19c37d;
  --accent-strong: #10a37f;
  --border: #13303f;
  --uploader-bg: #07101a;
  --input-bg: #0b1520;
  --input-text: #e6eef6;
  --shadow-sm: 0 6px 18px rgba(25, 195, 125, 0.06);
  --shadow-lg: 0 20px 50px rgba(25, 195, 125, 0.06);
}

/* App background */
[data-testid="stAppViewContainer"] { background: var(--bg); transition: background .25s ease; }

/* Force the sidebar visible */
[data-testid="stSidebar"], aside[role="complementary"],
[data-testid="stSidebar"][aria-expanded="false"], [data-testid="stSidebar"][aria-expanded="true"] {
  display:block !important;
  visibility:visible !important;
  transform:none !important;
  width:21rem !important;
  min-width:21rem !important;
  max-width:21rem !important;
  position:relative !important;
  left:0 !important;
  margin-left:0 !important;
  opacity:1 !important;
  z-index:9999 !important;
  background: var(--card) !important;
  border-right: 1px solid var(--border);
  transition: background .25s ease, border-color .25s ease;
}

/* Sidebar text colors */
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stMarkdown { color: var(--text) !important; }

/* Collapsed control visible */
[data-testid="collapsedControl"] { display:block !important; visibility:visible !important; color: var(--accent); background: var(--card); border:1px solid var(--border); border-radius:8px; padding:8px; z-index:10000 !important; }

/* Chat container */
.chat-container { max-width: 880px; margin: 0 auto; padding: 22px; color: var(--text); }

/* Message animations & bubbles */
@keyframes popSlide {
  0% { opacity: 0; transform: translateY(10px) scale(.995); filter: blur(4px); }
  60% { opacity: 1; transform: translateY(-4px) scale(1.01); filter: blur(0); }
  100% { transform: translateY(0) scale(1); filter: none; }
}
.message { display:flex; margin-bottom: 18px; animation: popSlide .36s cubic-bezier(.2,.9,.3,1) both; }
.message.user { justify-content: flex-end; }
.message-content {
  max-width: 82%;
  padding: 14px 18px;
  border-radius: 18px;
  line-height: 1.5;
  word-wrap: break-word;
  box-shadow: var(--shadow-sm);
  transition: transform .18s ease, box-shadow .18s ease;
}
.message.user .message-content {
  background: linear-gradient(180deg, var(--accent), var(--accent-strong));
  color: #fff;
  border-bottom-right-radius: 6px;
  transform-origin: right center;
}
.message.assistant .message-content {
  background: var(--card);
  color: var(--text);
  border: 1px solid var(--border);
  border-bottom-left-radius: 6px;
  transform-origin: left center;
}
.message .message-content:hover { transform: translateY(-2px); box-shadow: var(--shadow-lg); }

/* Avatar */
.avatar { width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:600; font-size:13px; margin:0 12px; flex-shrink:0; }
.message.user .avatar { background: var(--accent); color: #fff; order: 2; }
.message.assistant .avatar { background: var(--accent-strong); color: #fff; }

/* Capsule-style question box */
.stTextArea, .stTextArea > div { display:block !important; }
.stTextArea textarea {
  width:100% !important;
  min-height:88px !important;
  border-radius:999px !important;
  padding:18px 20px !important;
  font-size:15px !important;
  background: var(--input-bg) !important;
  color: var(--input-text) !important;
  border:1px solid var(--border) !important;
  box-shadow: var(--shadow-sm) !important;
  transition: box-shadow .18s ease, transform .12s ease, border-color .12s ease, background .18s ease;
  outline:none !important;
  resize:vertical !important;
}
.stTextArea textarea:focus {
  box-shadow: 0 10px 30px rgba(16,163,127,0.08);
  border-color: var(--accent);
  transform: translateY(-2px);
}

/* Send button */
.stButton > button {
  border-radius:999px !important;
  padding:12px 22px !important;
  background: linear-gradient(180deg, var(--accent), var(--accent-strong)) !important;
  color:#fff !important;
  border:none !important;
  box-shadow: 0 8px 26px rgba(16,163,127,0.12);
  transition: transform .12s ease, box-shadow .18s ease, opacity .12s ease;
}
.stButton > button:hover { transform: translateY(-3px); box-shadow: 0 18px 48px rgba(16,163,127,0.12); }

/* File uploader */
[data-testid="stFileUploader"] {
  background: var(--uploader-bg) !important;
  border-radius: 12px;
  padding: 18px;
  border: 1px dashed var(--border);
  box-shadow: var(--shadow-sm);
  transition: border-color .15s ease, box-shadow .15s ease, background .15s ease;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent); box-shadow: var(--shadow-lg); }

/* Status badge */
.status-badge { display:inline-block; padding:8px 16px; border-radius:20px; font-size:14px; font-weight:600; margin:8px 0; text-align:center; transition: transform .18s ease, box-shadow .18s ease; }
@keyframes pulseSoft { 0% { transform: scale(1); opacity: 1; } 60% { transform: scale(1.03); opacity: 0.98; } 100% { transform: scale(1); opacity: 1; } }
.status-ready { background: rgba(16,163,127,0.08); color: var(--accent-strong); border: 1px solid rgba(16,163,127,0.14); animation: pulseSoft 2.8s ease-in-out infinite; box-shadow: none; }
.status-not-ready { background: #fff6f6; color: #d63; border:1px solid #f7d6d6; }

/* Subtle transitions */
.stButton, .stTextArea, .stFileUploader, [data-testid="stSidebar"] { transition: all .18s ease; }

/* Hide Streamlit chrome (optional) */
#MainMenu, header, footer { visibility: hidden !important; height: 0 !important; padding: 0 !important; margin: 0 !important; }

/* Responsive */
@media (max-width: 780px) {
  .chat-container { padding: 14px; }
  [data-testid="stSidebar"] { width: 100% !important; max-width: 100% !important; position: relative !important; }
}
</style>
"""

# ---------------------------
# Main App
# ---------------------------
def main():
    st.markdown(MODERN_CSS, unsafe_allow_html=True)
    init_session_state()

    # -------------------------
    # SIDEBAR (theme toggle + controls)
    # -------------------------
    with st.sidebar:
        # Theme toggle (top of sidebar)
        st.session_state.dark_mode = st.checkbox(
            "🌙 Dark mode" if not st.session_state.dark_mode else "☀️ Light mode",
            value=st.session_state.dark_mode
        )

        st.markdown("## 📄 Documents")
        st.markdown("")
        # Status indicator
        if st.session_state.hnsw_ready:
            st.markdown('<div class="status-badge status-ready">✓ Documents Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-not-ready">⚠ Upload PDFs to Start</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📤 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload one or more PDF files to chat with"
        )

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
                    status_text.text("📖 Reading PDF files...")
                    progress_bar.progress(20)
                    raw_text = get_pdf_text(uploaded_files)
                    if not raw_text.strip():
                        st.error("❌ No readable text found in PDFs")
                    else:
                        status_text.text("✂️ Splitting text into chunks...")
                        progress_bar.progress(40)
                        text_chunks = get_text_chunks(raw_text)

                        status_text.text("🔍 Building search index...")
                        progress_bar.progress(60)

                        # Prefer local embeddings + hnsw if available
                        if HAVE_SENTENCE_TRANSFORMERS and HAVE_HNSWLIB:
                            try:
                                embeddings = LocalEmbeddings()
                                index = LocalHNSW.from_texts(text_chunks, embedding=embeddings, space='cosine')
                                status_text.text("💾 Saving index...")
                                progress_bar.progress(80)
                                index.save_local(HNSW_DIR)
                                st.session_state.use_local_hnsw = True
                                st.session_state.processed_files = [getattr(f, "name", "file") for f in uploaded_files]
                                st.session_state.hnsw_ready = True
                                st.success(f"✅ Successfully processed {len(text_chunks)} text chunks with local HNSW!")
                            except Exception as e:
                                # fallback if local build fails
                                st.warning("Local HNSW build failed — falling back to lexical retrieval. Install sentence-transformers & hnswlib for better results.")
                                st.error(str(e))
                                st.session_state.chunks = text_chunks
                                st.session_state.processed_files = [getattr(f, "name", "file") for f in uploaded_files]
                                st.session_state.use_local_hnsw = False
                                st.session_state.hnsw_ready = True
                        else:
                            # Lightweight fallback: store chunks for lexical retrieval
                            st.info("Using lexical retriever fallback (sentence-transformers or hnswlib not installed).")
                            st.session_state.chunks = text_chunks
                            st.session_state.processed_files = [getattr(f, "name", "file") for f in uploaded_files]
                            st.session_state.use_local_hnsw = False
                            st.session_state.hnsw_ready = True

                        progress_bar.progress(100)
                        status_text.empty()
                        progress_bar.empty()
                        time.sleep(0.6)
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()

        st.markdown("---")
        if st.session_state.messages:
            st.markdown(f"**💬 Messages:** {len(st.session_state.messages)}")

        st.markdown("---")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.experimental_rerun()

        if st.session_state.hnsw_ready:
            if st.button("❌ Delete Documents", use_container_width=True):
                try:
                    if os.path.isdir(HNSW_DIR):
                        shutil.rmtree(HNSW_DIR)
                    # clear session stored data
                    st.session_state.hnsw_ready = False
                    st.session_state.use_local_hnsw = False
                    st.session_state.chunks = []
                    st.session_state.processed_files = []
                    st.success("Documents deleted")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # Set theme attribute on <html> (must be OUTSIDE sidebar 'with' block)
    theme = "dark" if st.session_state.dark_mode else "light"
    st.markdown(
        f"""<script>
        try {{
          document.documentElement.setAttribute('data-theme', '{theme}');
        }} catch(e) {{ console.warn(e); }}
        </script>""",
        unsafe_allow_html=True,
    )

    # -------------------------
    # Main Chat Interface
    # -------------------------
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("ChatPDF")
        st.markdown('<p class="subtitle">Ask questions about your documents</p>', unsafe_allow_html=True)

    # Display messages or welcome
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
                escaped = html_module.escape(content)
                st.markdown(
                    f"""<div class='message user'><div class='message-content'>{escaped}</div><div class='avatar'>You</div></div>""",
                    unsafe_allow_html=True,
                )
            else:
                formatted_content = html_module.escape(content).replace("\n", "<br>")
                st.markdown(
                    f"""<div class='message assistant'><div class='avatar'>AI</div><div class='message-content'>{formatted_content}</div></div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # Input area at the bottom
    st.markdown("<br>", unsafe_allow_html=True)

    # Create input form (capsule + send button)
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
                st.session_state.messages.append({"role": "user", "content": user_input.strip()})

                with st.spinner("🤔 Searching documents and generating answer..."):
                    # Retrieve docs either via HNSW or lexical fallback
                    docs = []
                    if st.session_state.use_local_hnsw and HAVE_SENTENCE_TRANSFORMERS and HAVE_HNSWLIB:
                        try:
                            embeddings = LocalEmbeddings()
                            db = LocalHNSW.load_local(HNSW_DIR, embedding=embeddings)
                            docs = db.similarity_search(user_input, k=RETRIEVE_K, embedding=embeddings)
                        except Exception as e:
                            st.warning("Local HNSW lookup failed; falling back to lexical retriever.")
                            docs = retrieve_top_chunks_lexical(user_input, st.session_state.chunks, top_k=RETRIEVE_K)
                    else:
                        docs = retrieve_top_chunks_lexical(user_input, st.session_state.chunks, top_k=RETRIEVE_K)

                    # If we have docs, call generator; otherwise helpful message
                    if docs:
                        prompt_template = build_unified_prompt() if LANGCHAIN_AVAILABLE else None
                        google_api_key = st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
                        answer, model_used, error = generate_answer(prompt_template, docs, user_input, google_api_key=google_api_key)
                        if answer:
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            # if generation failed, show fallback summary from docs
                            concat = "\n\n".join([d["page_content"] if isinstance(d, dict) else d.page_content for d in docs])
                            fallback_excerpt = concat[:800].strip() + ("..." if len(concat) > 800 else "")
                            fallback_text = f"(No LLM available) Here's extracted context:\n\n{fallback_excerpt}"
                            st.session_state.messages.append({"role": "assistant", "content": fallback_text})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "I couldn't find any relevant information in the documents to answer your question."})

                st.experimental_rerun()

if __name__ == "__main__":
    main()
