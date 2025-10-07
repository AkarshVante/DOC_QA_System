
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

# ---------- Optional heavy imports (handle if missing) ----------
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

# LangChain + Google generative
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

# ---------- Config ----------
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

# ---------- Local Embeddings & HNSW ----------
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

# ---------- PDF Processing ----------
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
                # Fallback if PyPDF2 not installed
                text += f"[Could not extract {getattr(pdf,'name','file')} — install PyPDF2]\\n\\n"
        except Exception:
            st.warning(f"⚠️ Couldn't read {getattr(pdf,'name', 'a file')}")
            continue
    return text

# ---------- Text Chunking ----------
def get_text_chunks(text):
    # Split text into overlapping chunks
    text = text.replace("\n", " ")
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > CHUNK_SIZE:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-CHUNK_OVERLAP:]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ---------- Answer Generation ----------
def build_unified_prompt():
    # Prompt template for generative models (LangChain)
    template = PromptTemplate(
        input_variables=["input_documents", "question"],
        template=(
            "Use the following extracted parts of a PDF and answer the user question "
            "using only that information.\n\n{input_documents}\nQuestion: {question}"
        )
    )
    return template

def generate_answer(prompt_template, docs, question, google_api_key=None):
    """
    Tries LangChain + Gemini (if available with API key), else HF pipeline as fallback.
    """
    if not docs:
        return None, None, "No documents retrieved."

    context = "\n\n".join(
        [d.page_content if hasattr(d, "page_content") else d["page_content"] for d in docs]
    )

    # Gemini via LangChain if possible
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

    # Transformers pipeline fallback
    if HAVE_TRANSFORMERS:
        try:
            hf = pipeline("text2text-generation", model=HF_FALLBACK_MODEL, device=-1)
            prompt_text = (
                prompt_template.format(context=context, question=question)
                if prompt_template
                else f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            resp = hf(prompt_text, max_length=256, do_sample=False)
            if isinstance(resp, list) and resp:
                out = resp[0].get("generated_text") or resp[0].get("text") or str(resp[0])
                return clean_answer(out.strip()), HF_FALLBACK_MODEL, None
        except Exception as e:
            return None, None, f"Generation failed: {e}"

    # No model available
    return None, None, "No generation model available (no Gemini/HF)."

# ---------- Clean Answer Utility ----------
def clean_answer(text):
    """Clean up model response for presentation."""
    # Remove artifacts
    text = re.sub(r"content=(['\"])(.+?)\\1", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"\b(additional_kwargs|response_metadata|usage_metadata|id)=\{[^}]*\}", "", text, flags=re.DOTALL)

    # Remove markdown bold and trim
    text = text.replace("**", "").strip()

    # Remove leading bullets/numbers
    text = re.sub(r"^[\*\-\u2022]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[\.\)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\s*\n+", "\n\n", text, flags=re.MULTILINE)

    # Remove duplicate consecutive lines (keep single blank line)
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

# ---------- Lexical Retriever Fallback ----------
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
    top = [
        {"page_content": c, "metadata": {}}
        for i, score, c in scored[:top_k]
        if score > 0
    ]
    return top

# ---------- Session State Initialization ----------
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

# ---------- Modern UI CSS (capsule, animations, theme) ----------
MODERN_CSS = """
<style>
/* Fonts & base */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { box-sizing: border-box; font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; }

/* Theme color variables */
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

/* Always show sidebar */
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

/* Sidebar text color */
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stMarkdown { color: var(--text) !important; }

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
  background: var(--card);
  border: 2px solid #add8e6;  /* Light blue border for chat bubbles */
  color: var(--text);
}
.message-content:hover { transform: translateY(-2px); box-shadow: var(--shadow-lg); }

/* Avatars */
.avatar { width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:600; font-size:13px; margin:0 12px; flex-shrink:0; }
.message.user .avatar { background: var(--accent); color: #fff; order: 2; }
.message.assistant .avatar { background: var(--accent-strong); color: #fff; }

/* Input box (capsule style) */
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
  border: 2px solid var(--border);  /* Visible border for uploader area */
  box-shadow: var(--shadow-sm);
  transition: border-color .15s ease, box-shadow .15s ease, background .15s ease;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent); box-shadow: var(--shadow-lg); }

/* Status badge */
.status-badge { display:inline-block; padding:8px 16px; border-radius:20px; font-size:14px; font-weight:600; margin:8px 0; text-align:center; }
.status-ready { background: rgba(16,163,127,0.08); color: var(--accent-strong); border: 1px solid rgba(16,163,127,0.14); }
.status-not-ready { background: #fff6f6; color: #d63; border:1px solid #f7d6d6; }

/* Hide Streamlit menu/footer */
#MainMenu, header, footer { visibility: hidden !important; height: 0 !important; padding: 0 !important; margin: 0 !important; }

/* Responsive */
@media (max-width: 780px) {
  .chat-container { padding: 14px; }
  [data-testid="stSidebar"] { width: 100% !important; max-width: 100% !important; position: relative !important; }
}
</style>
"""

def main():
    # Inject custom CSS for theming and modern UI
    st.markdown(MODERN_CSS, unsafe_allow_html=True)
    init_session_state()

    # Apply dark/light theme toggle
    theme = "dark" if st.session_state.dark_mode else "light"
    st.markdown(
        f"<script>document.documentElement.setAttribute('data-theme', '{theme}');</script>",
        unsafe_allow_html=True
    )

    # ---------- Sidebar (theme toggle + document controls) ----------
    with st.sidebar:
        # Theme toggle checkbox
        st.session_state.dark_mode = st.checkbox(
            "🌙 Dark mode" if not st.session_state.dark_mode else "☀️ Light mode",
            value=st.session_state.dark_mode
        )

        st.markdown("## 📄 Documents")
        st.markdown("")
        # Documents status badge
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

        if uploaded_files and st.button("🔄 Process Documents", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                status_text.text("📖 Reading PDF files...")
                progress_bar.progress(20)
                raw_text = get_pdf_text(uploaded_files)
                if not raw_text.strip():
                    st.error("❌ No readable text found in PDFs. Ensure the PDF contains selectable text.")
                else:
                    status_text.text("✂️ Splitting text into chunks...")
                    progress_bar.progress(40)
                    text_chunks = get_text_chunks(raw_text)

                    status_text.text("🔍 Building search index...")
                    progress_bar.progress(60)

                    # Try local embeddings + HNSW
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
                            st.success(f"✅ Processed {len(text_chunks)} text chunks with HNSW!")
                        except Exception as e:
                            st.warning("Local HNSW build failed — falling back to lexical retrieval.")
                            st.error(f"❌ Error: {e}")
                            st.session_state.chunks = text_chunks
                            st.session_state.processed_files = [getattr(f, "name", "file") for f in uploaded_files]
                            st.session_state.use_local_hnsw = False
                            st.session_state.hnsw_ready = True
                    else:
                        # Fallback to lexical
                        st.info("Using lexical retriever fallback (no sentence-transformers/hnswlib).")
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
                st.error(f"❌ Error processing documents: {e}. Please check your PDFs and try again.")
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
                    st.session_state.hnsw_ready = False
                    st.session_state.use_local_hnsw = False
                    st.session_state.chunks = []
                    st.session_state.processed_files = []
                    st.success("Documents deleted")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # ---------- Main Chat Interface ----------
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Centered title with icon
    st.markdown("<h1 style='text-align: center;'>📄 ChatPDF</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle" style="text-align: center;">Ask questions about your documents</p>', unsafe_allow_html=True)

    # Welcome or instructions
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
                    f"<div class='message user'><div class='message-content'>{escaped}</div><div class='avatar'>🙂</div></div>",
                    unsafe_allow_html=True
                )
            else:
                formatted_content = html_module.escape(content).replace("\n", "<br>")
                st.markdown(
                    f"<div class='message assistant'><div class='avatar'>🤖</div><div class='message-content'>{formatted_content}</div></div>",
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # Input form at bottom
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_area(
                "Message",
                placeholder=(
                    "Ask a question about your documents... (Ctrl+Enter to send)"
                    if st.session_state.hnsw_ready
                    else "Upload and process PDFs first..."
                ),
                height=100,
                label_visibility="collapsed",
                key="user_input",
                disabled=not st.session_state.hnsw_ready
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit_disabled = not st.session_state.hnsw_ready
            submit = st.form_submit_button("Send", use_container_width=True, disabled=submit_disabled)

        if submit:
            if not st.session_state.hnsw_ready:
                st.error("⚠️ Please upload and process PDF documents before asking a question.")
            elif user_input and user_input.strip():
                # Append user question
                st.session_state.messages.append({"role": "user", "content": user_input.strip()})

                with st.spinner("🤔 Searching documents and generating answer..."):
                    # Retrieve documents via HNSW or lexical
                    if st.session_state.use_local_hnsw and HAVE_SENTENCE_TRANSFORMERS and HAVE_HNSWLIB:
                        try:
                            embeddings = LocalEmbeddings()
                            db = LocalHNSW.load_local(HNSW_DIR, embedding=embeddings)
                            docs = db.similarity_search(user_input, k=RETRIEVE_K, embedding=embeddings)
                        except Exception:
                            st.warning("Local HNSW lookup failed; using lexical retriever.")
                            docs = retrieve_top_chunks_lexical(user_input, st.session_state.chunks, top_k=RETRIEVE_K)
                    else:
                        docs = retrieve_top_chunks_lexical(user_input, st.session_state.chunks, top_k=RETRIEVE_K)

                    if docs:
                        prompt_template = build_unified_prompt() if LANGCHAIN_AVAILABLE else None
                        google_api_key = (
                            st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
                        )
                        answer, model_used, error = generate_answer(prompt_template, docs, user_input, google_api_key=google_api_key)
                        if answer:
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            # No LLM available: show context instead
                            concat = "\n\n".join(
                                [d["page_content"] if isinstance(d, dict) else d.page_content for d in docs]
                            )
                            excerpt = (concat[:800].strip() + "...") if len(concat) > 800 else concat
                            fallback_text = f"(No LLM available) Extracted context:\n\n{excerpt}"
                            st.session_state.messages.append({"role": "assistant", "content": fallback_text})
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "⚠️ I couldn't find relevant information in the documents to answer your question. Please try rephrasing or check your PDFs."
                        })

                st.experimental_rerun()

if __name__ == "__main__":
    main()
