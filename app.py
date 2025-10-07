# app.py
"""
ChatPDF - Streamlit app

Changes in this version:
- Fixed double-click send problem by using a single `st.button` + processing flag.
- Removed theme toggle (no light/dark switch).
- Locked sidebar visible and removed sidebar toggle.
- Added thin light-blue border to sidebar buttons.
- Kept full functionality: PDF processing, chunking, HNSW (if available), lexical fallback,
  and answer generation with Gemini/transformers fallback.
- Improved error messages and defensive checks.
"""

import os
import time
import shutil
import pickle
import html as html_module
import re
from io import BytesIO

import streamlit as st
import numpy as np

# ---------- Optional dependencies (handled gracefully if missing) ----------
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

# ---------- App configuration ----------
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

# ---------- Local embedding & HNSW helper classes ----------
class LocalEmbeddings:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        if not HAVE_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers not installed")
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
        self.index.save_index(os.path.join(folder, self.INDEX_FILENAME))
        with open(os.path.join(folder, self.META_FILENAME), "wb") as f:
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
                    from langchain.schema import Document as LC_Doc
                    results.append(LC_Doc(page_content=rec["text"], metadata={"score": float(dist)}))
                else:
                    results.append({"page_content": rec["text"], "metadata": {"score": float(dist)}})
        return results

# ---------- PDF processing ----------
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
                text += f"[Could not extract {getattr(pdf,'name','file')} — install PyPDF2 to enable extraction]\n\n"
        except Exception as e:
            st.warning(f"⚠️ Couldn't read {getattr(pdf,'name','a file')}: {e}")
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

# ---------- Generator (LangChain / HF fallback) ----------
def build_unified_prompt():
    template = """You are a helpful AI assistant that answers questions based on the provided context.

Instructions:
- Provide a clear, conversational answer based ONLY on the context below
- If the answer is not in the context, say "I don't have enough information in the provided documents to answer that question."
- Keep the response concise.

Context:
{context}

Question: {question}

Answer:"""
    return PromptTemplate(template=template, input_variables=["context", "question"]) if LANGCHAIN_AVAILABLE else None

def generate_answer(prompt_template, docs, question, google_api_key=None):
    if not docs:
        return None, None, "No documents retrieved."

    context = "\n\n".join([d.page_content if hasattr(d, "page_content") else d["page_content"] for d in docs])

    # Try Gemini via LangChain
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

    # Transformers fallback
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

    return None, None, "No generation model available (no Gemini/HF)."

def clean_answer(text):
    text = re.sub(r"content=(['\"])(.+?)\1", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"\b(additional_kwargs|response_metadata|usage_metadata|id)=\{[^}]*\}", "", text, flags=re.DOTALL)
    text = text.replace("**", "").strip()
    text = re.sub(r"^[\*\-\u2022]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[\.\)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\s*\n+", "\n\n", text, flags=re.MULTILINE)

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

# ---------- Lexical fallback ----------
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
    top = [{"page_content": c, "metadata": {}} for i, score, c in scored[:top_k] if score > 0]
    return top

# ---------- Session state initialization ----------
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
    if "processing" not in st.session_state:
        st.session_state.processing = False
    # Keep the text input value in session_state so we can clear it reliably
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

# ---------- CSS (locked sidebar, light-blue borders for sidebar buttons) ----------
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { box-sizing: border-box; font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial; }

/* Basic color vars (single light theme) */
:root {
  --bg: #f5f7f9;
  --card: #ffffff;
  --text: #0f1720;
  --muted: #6b7280;
  --accent: #10a37f;
  --border: #e6e9ee;
  --uploader-bg: #ffffff;
  --input-bg: #ffffff;
  --input-text: #0f1720;
  --shadow-sm: 0 6px 18px rgba(16, 163, 127, 0.06);
}

/* App background */
[data-testid="stAppViewContainer"] { background: var(--bg); transition: background .25s ease; }

/* Force sidebar visible & lock it (no collapsed control) */
[data-testid="stSidebar"], aside[role="complementary"] {
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
}
/* hide the small collapsed control entirely */
[data-testid="collapsedControl"] { display:none !important; visibility:hidden !important; }

/* Sidebar button thin light-blue border */
[data-testid="stSidebar"] .stButton > button {
  border: 1px solid #add8e6 !important;
  background: transparent !important;
  color: var(--text) !important;
  box-shadow: none !important;
  border-radius: 8px !important;
  padding: 8px 12px !important;
}

/* Chat container */
.chat-container { max-width: 880px; margin: 0 auto; padding: 22px; color: var(--text); }

/* Message bubbles & animation */
@keyframes popSlide { 0% { opacity: 0; transform: translateY(10px) scale(.995); } 60% { opacity:1; transform: translateY(-4px) scale(1.01);} 100% { transform:translateY(0) scale(1); } }
.message { display:flex; margin-bottom: 18px; animation: popSlide .36s cubic-bezier(.2,.9,.3,1) both; }
.message.user { justify-content:flex-end; }
.message-content {
  max-width:82%; padding:14px 18px; border-radius:14px; line-height:1.5; word-wrap:break-word;
  box-shadow: var(--shadow-sm); transition: transform .12s ease, box-shadow .12s ease;
  background: var(--card); border: 2px solid #add8e6; color: var(--text);
}
.message-content:hover { transform: translateY(-2px); box-shadow: 0 12px 30px rgba(16,163,127,0.06); }

/* Avatar */
.avatar { width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:600; font-size:14px; margin:0 12px; flex-shrink:0; background:#10a37f; color:#fff; }

/* Input */
.stTextArea textarea { width:100% !important; min-height:84px !important; border-radius:999px !important; padding:14px 18px !important; font-size:15px !important; background: var(--input-bg) !important; color: var(--input-text) !important; border:1px solid var(--border) !important; box-shadow: var(--shadow-sm) !important; transition: box-shadow .12s ease, transform .08s ease; outline:none !important; resize:vertical !important; }
.stTextArea textarea:focus { box-shadow: 0 10px 30px rgba(16,163,127,0.08); border-color: var(--accent); transform: translateY(-1px); }

/* File uploader box */
[data-testid="stFileUploader"] {
  background: var(--uploader-bg) !important;
  border-radius: 12px;
  padding: 18px;
  border: 2px dashed #add8e6 !important;
  box-shadow: var(--shadow-sm);
  transition: border-color .15s ease, box-shadow .15s ease, background .15s ease;
}

/* Hide Streamlit chrome */
#MainMenu, header, footer { visibility: hidden !important; height: 0 !important; padding: 0 !important; margin: 0 !important; }

@media (max-width:780px) {
  .chat-container { padding: 14px; }
  [data-testid="stSidebar"] { width:100% !important; max-width:100% !important; }
}
</style>
"""

# ---------- Main app ----------
def main():
    st.markdown(APP_CSS, unsafe_allow_html=True)
    init_session_state()

    # ------------- SIDEBAR (locked, no toggle) -------------
    with st.sidebar:
        st.markdown("## 📄 Documents")
        if st.session_state.hnsw_ready:
            st.markdown('<div style="padding:6px;border-radius:8px;background:rgba(16,163,127,0.06);border:1px solid rgba(16,163,127,0.14);font-weight:600;">✓ Documents Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="padding:6px;border-radius:8px;background:#fff6f6;border:1px solid #f7d6d6;font-weight:600;">⚠ Upload PDFs to Start</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📤 Upload PDFs")
        uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=['pdf'], help="Upload one or more PDF files to chat with")

        if uploaded_files:
            st.markdown("**Uploaded files:**")
            for f in uploaded_files:
                st.text(f"📄 {f.name}")

        if uploaded_files and st.button("🔄 Process Documents", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            try:
                status.text("📖 Reading PDF files...")
                progress.progress(10)
                raw_text = get_pdf_text(uploaded_files)
                if not raw_text.strip():
                    st.error("❌ No readable text found in the provided PDFs. If they are scans, run OCR or upload text-based PDFs.")
                    progress.empty()
                    status.empty()
                else:
                    status.text("✂️ Splitting text into chunks...")
                    progress.progress(40)
                    chunks = get_text_chunks(raw_text)
                    status.text("🔍 Building search index...")
                    progress.progress(60)

                    if HAVE_SENTENCE_TRANSFORMERS and HAVE_HNSWLIB:
                        try:
                            embeddings = LocalEmbeddings()
                            index = LocalHNSW.from_texts(chunks, embedding=embeddings, space='cosine')
                            status.text("💾 Saving index...")
                            progress.progress(80)
                            index.save_local(HNSW_DIR)
                            st.session_state.use_local_hnsw = True
                            st.session_state.chunks = chunks
                            st.session_state.processed_files = [getattr(f, "name", "file") for f in uploaded_files]
                            st.session_state.hnsw_ready = True
                            st.success("✅ Documents processed and HNSW index saved.")
                            progress.progress(100)
                        except Exception as e:
                            st.warning("Local HNSW build failed — falling back to lexical retrieval.")
                            st.error(f"Error building HNSW index: {e}")
                            st.session_state.chunks = chunks
                            st.session_state.use_local_hnsw = False
                            st.session_state.hnsw_ready = True
                    else:
                        st.info("Using lexical retriever fallback (sentence-transformers or hnswlib not installed).")
                        st.session_state.chunks = chunks
                        st.session_state.use_local_hnsw = False
                        st.session_state.hnsw_ready = True

                    progress.empty()
                    status.empty()
            except Exception as e:
                st.error(f"❌ Error processing documents: {e}. Please check the PDF files.")
                progress.empty()
                status.empty()

        st.markdown("---")
        if st.session_state.messages:
            st.markdown(f"**💬 Messages:** {len(st.session_state.messages)}")
        st.markdown("---")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
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
                except Exception as e:
                    st.error(f"Error deleting documents: {e}")

    # ------------- Main UI -------------
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;margin-bottom:6px;'>📄 ChatPDF</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;margin-top:0;color:#6b7280'>Ask questions about your documents</p>", unsafe_allow_html=True)

    # Show welcome or chat messages
    if not st.session_state.messages:
        if st.session_state.hnsw_ready:
            st.markdown("<div style='padding:12px;border-radius:10px;border:1px solid var(--border);background:var(--card)'>✅ Documents ready — ask a question below.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding:12px;border-radius:10px;border:1px solid var(--border);background:var(--card)'>👋 Upload PDF(s) from the sidebar to begin.</div>", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            rendered = html_module.escape(content).replace("\n", "<br>")
            if role == "user":
                st.markdown(f"<div class='message user'><div class='message-content'>{rendered}</div><div class='avatar'>You</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='message assistant'><div class='avatar'>AI</div><div class='message-content'>{rendered}</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ------------- Input area (NO form) -------------
    # Keep the text in session_state so we can control it reliably
    user_text = st.text_area("Message", value=st.session_state.get("user_input", ""), height=100,
                             placeholder=("Ask about your documents..." if st.session_state.hnsw_ready else "Upload & process PDFs first..."),
                             disabled=not st.session_state.hnsw_ready, key="user_input")

    # Single Send button; processing flag prevents double-submits
    send_disabled = not st.session_state.hnsw_ready or st.session_state.processing
    if st.button("Send", disabled=send_disabled, use_container_width=False):
        # Defensive checks
        if not st.session_state.hnsw_ready:
            st.error("⚠️ Upload and process PDFs from the sidebar before asking questions.")
        elif not user_text or not user_text.strip():
            st.warning("Please enter a question before clicking Send.")
        elif st.session_state.processing:
            # unlikely due to button disabling, but safe guard
            st.info("Processing already in progress, please wait...")
        else:
            # Mark processing to avoid re-entrancy / double click
            st.session_state.processing = True

            # Append user message
            q = user_text.strip()
            st.session_state.messages.append({"role": "user", "content": q})

            # Perform retrieval + generation synchronously
            with st.spinner("🤔 Retrieving relevant parts and generating an answer..."):
                # Retrieve docs using HNSW or lexical fallback
                docs = []
                if st.session_state.use_local_hnsw and HAVE_SENTENCE_TRANSFORMERS and HAVE_HNSWLIB:
                    try:
                        embeddings = LocalEmbeddings()
                        db = LocalHNSW.load_local(HNSW_DIR, embedding=embeddings)
                        docs = db.similarity_search(q, k=RETRIEVE_K, embedding=embeddings)
                    except Exception:
                        st.warning("HNSW retrieval failed; falling back to lexical retrieval.")
                        docs = retrieve_top_chunks_lexical(q, st.session_state.chunks, top_k=RETRIEVE_K)
                else:
                    docs = retrieve_top_chunks_lexical(q, st.session_state.chunks, top_k=RETRIEVE_K)

                # If no docs -> helpful assistant message
                if not docs:
                    st.session_state.messages.append({"role": "assistant",
                                                      "content": "⚠️ I couldn't find relevant information in your documents. Try rephrasing the question or ensure correct PDFs were uploaded."})
                else:
                    prompt_template = build_unified_prompt() if LANGCHAIN_AVAILABLE else None
                    google_api_key = (st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY"))
                    answer, model_used, error = generate_answer(prompt_template, docs, q, google_api_key=google_api_key)
                    if answer:
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        # Fallback: show concise extracted excerpt
                        concat = "\n\n".join([d["page_content"] if isinstance(d, dict) else d.page_content for d in docs])
                        excerpt = (concat[:1200].strip() + "...") if len(concat) > 1200 else concat
                        fallback_text = "(No LLM available) Extracted context from documents:\n\n" + excerpt
                        st.session_state.messages.append({"role": "assistant", "content": fallback_text})

            # Clear input and release processing flag
            st.session_state.user_input = ""
            st.session_state.processing = False

            # Note: no experimental_rerun(); Streamlit reruns automatically after user interaction,
            # and session_state changes persist so the updated messages will render immediately.

if __name__ == "__main__":
    main()
