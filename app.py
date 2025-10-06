# app.py
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

# hnswlib for vector index (robust on Streamlit Cloud)
import hnswlib

# LangChain types used for QA chain option and Documents
try:
    from langchain.schema import Document
    from langchain.prompts import PromptTemplate
    from langchain.chains.question_answering import load_qa_chain
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except Exception:
    # LangChain or Google GenAI integration may be missing — we'll still run fallback generation.
    LANGCHAIN_AVAILABLE = False

# HuggingFace fallback pipeline for generation
from transformers import pipeline
# Ensure CPU fallback by default
HF_FALLBACK_MODEL = "google/flan-t5-small"

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="ChatPDF — Green & Black", layout="wide")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers model id
HNSW_DIR = "hnsw_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVE_K = 4

# Which Gemini model names to try (order)
GEMINI_PREFERRED = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

# ---------------------------
# Helpers: embeddings wrapper
# ---------------------------
class LocalEmbeddings:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        vectors = self.model.encode(texts, show_progress_bar=False)
        # return list[list[float]]
        return [vec.tolist() if hasattr(vec, "tolist") else list(map(float, vec)) for vec in vectors]

    def embed_query(self, text):
        vec = self.model.encode([text], show_progress_bar=False)[0]
        return vec.tolist() if hasattr(vec, "tolist") else list(map(float, vec))

# ---------------------------
# HNSW wrapper
# ---------------------------
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
                # Use langchain Document if available otherwise a simple dict
                if LANGCHAIN_AVAILABLE:
                    results.append(Document(page_content=rec["text"], metadata={"score": float(dist)}))
                else:
                    results.append({"page_content": rec["text"], "metadata": {"score": float(dist)}})
        return results

# ---------------------------
# PDF -> text -> chunks
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
            # skip unreadable files but show small note
            st.warning(f"Warning: couldn't read a file: {getattr(pdf,'name', str(pdf))} ({e})")
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
# Prompt templates
# ---------------------------
def build_plain_prompt():
    template = (
        """
You are an assistant that answers using ONLY the provided context.

Provide a concise single-line direct answer first. If the answer is not present in the context, respond exactly with:
Answer is not available in the context.

After that, provide a short explanation in 1-3 sentences. Keep paragraphs short.

Context:
{context}

Question:
{question}
"""
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])

def build_bullets_prompt():
    template = (
        """
You are an assistant that answers using ONLY the provided context.

Start with a single-line direct answer (or "Answer is not available in the context."). Then include a 'Key points:' section with 3-6 bullet points listing the important facts or steps. Use concise bullets.

Context:
{context}

Question:
{question}
"""
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ---------------------------
# Generate answer (try Gemini via LangChain first, then HF fallback)
# ---------------------------
def generate_answer(prompt_template, docs, question, google_api_key=None):
    """
    Returns (answer_text, model_name, error_or_none)
    Tries Gemini via langchain_google_genai (if installed and api key provided),
    else falls back to local HF pipeline.
    """
    # Build context text from docs
    if not docs:
        return None, None, "No docs for retrieval."
    # convert docs to a single context string (limit size if needed)
    context = "\n\n".join([d.page_content if hasattr(d, "page_content") else d["page_content"] for d in docs])
    prompt_text = prompt_template.format(context=context, question=question)

    # Attempt Gemini via LangChain + ChatGoogleGenerativeAI if available and key provided
    if LANGCHAIN_AVAILABLE and google_api_key:
        for model_name in GEMINI_PREFERRED:
            try:
                model = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
                chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)
                response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
                text = None
                if isinstance(response, dict):
                    for key in ("output_text", "text", "answer", "output"):
                        if key in response and response[key]:
                            text = response[key]
                            break
                    if not text:
                        for v in response.values():
                            if isinstance(v, str) and v.strip():
                                text = v
                                break
                            if isinstance(v, list) and v:
                                parts = [p for p in v if isinstance(p, str) and p.strip()]
                                if parts:
                                    text = "\n".join(parts)
                                    break
                elif isinstance(response, str):
                    text = response
                else:
                    try:
                        text = str(response)
                    except Exception:
                        text = None

                if text and text.strip():
                    return text.strip(), model_name, None
                else:
                    continue
            except Exception as e:
                # try next model
                continue

    # Fallback to HuggingFace pipeline
    try:
        hf = pipeline("text2text-generation", model=HF_FALLBACK_MODEL, device=-1)
        # Use the assembled prompt_text
        resp = hf(prompt_text, max_length=256, do_sample=False)
        if isinstance(resp, list) and resp:
            out = resp[0].get("generated_text") or resp[0].get("text") or str(resp[0])
            return out.strip(), HF_FALLBACK_MODEL, None
        else:
            return None, None, "HF model returned empty."
    except Exception as e:
        return None, None, f"Generation failed: {e}"

# ---------------------------
# Formatting helpers
# ---------------------------
def format_to_bullets(text: str) -> str:
    # Clean and convert into markdown bullets
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return text
    # If it's already a single paragraph, try split by sentences.
    if len(lines) == 1 and len(lines[0]) > 120:
        # naive sentence split
        import re
        sents = re.split(r'(?<=[.!?])\s+', lines[0])
        lines = [s.strip() for s in sents if s.strip()]
    bullets = "\n".join([f"- {html_module.escape(line)}" for line in lines])
    return bullets

# ---------------------------
# Session state helpers
# ---------------------------
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "hnsw_ready" not in st.session_state:
        st.session_state.hnsw_ready = os.path.isdir(HNSW_DIR)
    if "last_model_used" not in st.session_state:
        st.session_state.last_model_used = None
    if "focus_index" not in st.session_state:
        st.session_state.focus_index = None

def add_message(role, text):
    st.session_state.history.append({
        "role": role,
        "text": text,
        "time": datetime.datetime.now().isoformat()
    })

def find_preceding_user_message_text(idx):
    for i in range(idx - 1, -1, -1):
        if st.session_state.history[i]["role"] == "user":
            return st.session_state.history[i]["text"]
    return None

def format_time(iso_ts: str) -> str:
    try:
        dt = datetime.datetime.fromisoformat(iso_ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            cleaned = iso_ts.rstrip("Z")
            dt = datetime.datetime.fromisoformat(cleaned)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return iso_ts

# ---------------------------
# UI: green + black theme CSS
# ---------------------------
CHAT_CSS = """
<style>
/* page background */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #0b0f0b 0%, #081008 100%);
  color: #b7f2b7;
}

/* sidebar background */
[data-testid="stSidebar"] {
  background: #061006;
  color: #b7f2b7;
}

/* titles and headings */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {
  color: #e6ffe6;
}

/* chat window */
.chat-window { max-height: 68vh; overflow: auto; padding: 12px; background: #07110a; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.6); }

/* messages */
.message { margin: 10px 0; display: flex; align-items: flex-end; }
.bubble { max-width: 78%; padding: 12px 14px; border-radius: 12px; line-height: 1.45; font-family: 'Inter', sans-serif; color: #e6ffe6; }
.bubble.user { background: linear-gradient(90deg,#0fa75d,#0a8a45); color: #ffffff; border-bottom-right-radius: 6px; margin-left: auto; }
.bubble.assistant { background: #0f2a19; color: #bff3b0; border-bottom-left-radius: 6px; margin-right: auto; }

/* meta */
.meta { font-size: 11px; color: #9dd59d; margin-top: 6px; }

/* focused */
.focused { box-shadow: 0 0 0 3px rgba(6, 95, 57, 0.18); border: 1px solid rgba(0,255,0,0.08); padding: 10px; background: linear-gradient(90deg, #07110a, #07210c); }

/* lists inside bubbles */
.bubble ul { margin: 8px 0 8px 18px; color: #dfffd8; }
.bubble ol { margin: 8px 0 8px 18px; color: #dfffd8; }
pre { background: #001209; color: #bff3b0; padding: 10px; border-radius: 8px; overflow: auto; }
</style>
"""

# ---------------------------
# Main app
# ---------------------------
def main():
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    init_session_state()

    # Header
    cols = st.columns([0.7, 0.3])
    with cols[0]:
        st.title("📄 ChatPDF — Green & Black")
        st.caption("Upload PDFs, process them locally, and chat. Bulleted, clean answers. No paid embedding APIs required.")
    with cols[1]:
        if st.session_state.hnsw_ready:
            st.success("HNSW index available.")
        else:
            st.info("No index yet — upload PDFs to build one.")

    left_col, center_col, right_col = st.columns([1.5, 3, 1])

    # Left: history + export
    with left_col:
        st.header("💬 Conversations")
        if not st.session_state.history:
            st.info("No messages yet — your conversation history will appear here.")
        else:
            preview_items = []
            for i, m in enumerate(st.session_state.history):
                role = "You" if m['role'] == 'user' else "Assistant"
                preview = m['text'][:90].replace("\n", " ")
                ts = format_time(m.get('time', ''))
                preview_items.append(f"{i}: [{ts}] {role}: {preview}")

            default_index = 0
            if st.session_state.get("focus_index") is not None:
                fi = st.session_state.focus_index
                if 0 <= fi < len(preview_items):
                    default_index = fi

            selected = st.radio("Select message to focus", options=list(range(len(preview_items))), index=default_index, format_func=lambda x: preview_items[x]) if preview_items else None
            if preview_items and st.session_state.get("focus_index") != selected:
                st.session_state.focus_index = selected

            cols_left_actions = st.columns([0.5, 0.5])
            with cols_left_actions[0]:
                if st.button("Clear History"):
                    st.session_state.history = []
                    st.session_state.focus_index = None
                    st.success("Chat history cleared.")
            with cols_left_actions[1]:
                st.write(" ")

            st.markdown("---")
            st.subheader("Export")
            conv_text = []
            for m in st.session_state.history:
                role = "You" if m['role'] == 'user' else 'Assistant'
                conv_text.append(f"[{format_time(m.get('time',''))}] {role}: {m['text']}")
            if conv_text:
                conv_joined = "\n".join(conv_text)
                st.download_button("Download conversation", data=conv_joined, file_name="chatpdf_conversation.txt")
            else:
                st.info("Nothing to download yet.")

    # Center: main chat
    with center_col:
        with st.form(key="question_form", clear_on_submit=True):
            user_question = st.text_area("", placeholder="Type your question and press Send (Shift+Enter for newline)...", height=120, key='chat_input')
            format_choice = st.selectbox("Answer style", options=["Plain text", "Bullets"], index=1)
            cols_btn = st.columns([0.18, 0.82])
            with cols_btn[0]:
                send_btn = st.form_submit_button("Send")
            with cols_btn[1]:
                st.write(" ")

            if send_btn and user_question and user_question.strip():
                if not st.session_state.hnsw_ready:
                    st.error("Index not found. Please upload & process PDFs first (use Upload PDFs below).")
                else:
                    add_message('user', user_question)
                    try:
                        embeddings = LocalEmbeddings()
                        new_db = LocalHNSW.load_local(HNSW_DIR, embedding=embeddings)
                        docs = new_db.similarity_search(user_question, k=RETRIEVE_K, embedding=embeddings)
                    except Exception as e:
                        st.error(f"Failed to load index: {e}")
                        docs = []

                    if docs:
                        with st.spinner("Generating answer (trying best available model)..."):
                            prompt_template = build_plain_prompt() if format_choice == "Plain text" else build_bullets_prompt()
                            google_api_key = st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
                            answer_text, model_used, error = generate_answer(prompt_template, docs, user_question, google_api_key=google_api_key)

                        if answer_text:
                            # Format to bullets if chosen or if user picked 'Bullets'
                            if format_choice == "Bullets":
                                formatted_answer = format_to_bullets(answer_text)
                            else:
                                formatted_answer = html_module.escape(answer_text)

                            add_message('assistant', formatted_answer)
                            st.session_state.last_model_used = model_used
                            st.session_state.focus_index = len(st.session_state.history) - 1
                            st.success("Answer generated and appended to conversation.")
                        else:
                            st.error(f"Failed to generate answer: {error}")

        # Upload and process PDFs
        with st.expander("📎 Upload PDFs (attach & process here)"):
            uploaded = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])
            if st.button("Submit & Process Files"):
                if not uploaded:
                    st.warning("Please upload one or more PDF files first.")
                else:
                    progress_text = st.empty()
                    progress_bar = st.progress(0)

                    progress_text.info("Stage 1/4 — Extracting text from PDFs...")
                    time.sleep(0.2)
                    raw_text = get_pdf_text(uploaded)
                    progress_bar.progress(10)

                    if not raw_text.strip():
                        st.error("No readable text found in uploaded PDFs.")
                    else:
                        progress_text.info("Stage 2/4 — Chunking text for embeddings...")
                        text_chunks = get_text_chunks(raw_text)
                        progress_bar.progress(40)

                        progress_text.info("Stage 3/4 — Creating embeddings and building HNSW index...")
                        def cb(msg):
                            progress_text.info(msg)

                        try:
                            embeddings = LocalEmbeddings()
                            index = LocalHNSW.from_texts(text_chunks, embedding=embeddings, space='cosine')
                            index.save_local(HNSW_DIR)
                            progress_bar.progress(90)
                            st.session_state.hnsw_ready = True
                            st.success(f"✅ Processing complete — indexed {len(text_chunks)} chunks.")
                        except Exception as e:
                            st.error(f"Failed to build index: {e}")
                            st.session_state.hnsw_ready = False

                        progress_bar.progress(100)

        # Render chat window
        chat_box = st.container()
        st.markdown("<div class='chat-window' id='chat-window'>", unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown("<div style='padding:20px;color:#9dd59d'>No messages yet — upload PDFs and ask a question!</div>", unsafe_allow_html=True)
        else:
            for idx, msg in enumerate(st.session_state.history):
                ts = format_time(msg.get('time',''))
                msg_id = f"msg-{idx}"
                focused_class = "focused" if st.session_state.focus_index is not None and st.session_state.focus_index == idx else ""
                if msg['role'] == 'assistant':
                    content_html = msg['text']
                    # If it's plain escaped text we want to keep line breaks
                    # Already escaped when added; show as markdown inside bubble
                    chat_box.markdown(
                        f"""
                        <div class='message' id='{msg_id}'>
                          <div class='bubble assistant {focused_class}'>{content_html}<div class='meta'>{ts}</div></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    content_html = "<p style='margin:0;color:#07110a'>" + html_module.escape(msg['text']).replace("\n","<br/>") + "</p>"
                    chat_box.markdown(
                        f"""
                        <div class='message' id='{msg_id}'>
                          <div class='bubble user {focused_class}'>{content_html}<div class='meta'>{ts}</div></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        st.markdown("</div>", unsafe_allow_html=True)

        # Scroll to focused message if set
        if st.session_state.focus_index is not None:
            focus_idx = st.session_state.focus_index
            if 0 <= focus_idx < len(st.session_state.history):
                scroll_script = f"""
                <script>
                const el = document.getElementById("msg-{focus_idx}");
                if (el) {{
                    el.scrollIntoView({{behavior: "smooth", block: "center"}});
                }}
                </script>
                """
                st.components.v1.html(scroll_script, height=0, width=0)

    # Right: controls
    with right_col:
        st.header("Controls")
        st.markdown("**Index status:**")
        if st.session_state.hnsw_ready:
            st.success("HNSW index available.")
        else:
            st.warning("No HNSW index found. Upload PDFs to build index.")

        st.markdown("---")
        st.subheader("Reformat Answer")
        focus_idx = st.session_state.focus_index
        target_idx = None
        if focus_idx is not None and 0 <= focus_idx < len(st.session_state.history):
            if st.session_state.history[focus_idx]["role"] == "assistant":
                target_idx = focus_idx
            else:
                for j in range(focus_idx+1, len(st.session_state.history)):
                    if st.session_state.history[j]["role"] == "assistant":
                        target_idx = j
                        break
        else:
            for j in range(len(st.session_state.history)-1, -1, -1):
                if st.session_state.history[j]["role"] == "assistant":
                    target_idx = j
                    break

        if target_idx is not None:
            st.write(f"Selected assistant message index: {target_idx}")
            cols_regen = st.columns([1,1])
            with cols_regen[0]:
                if st.button("Regenerate — Plain Text", key=f"regen_plain_{target_idx}"):
                    user_q = find_preceding_user_message_text(target_idx)
                    if not user_q:
                        st.error("Could not find the original user question.")
                    else:
                        try:
                            embeddings = LocalEmbeddings()
                            new_db = LocalHNSW.load_local(HNSW_DIR, embedding=embeddings)
                            docs = new_db.similarity_search(user_q, k=RETRIEVE_K, embedding=embeddings)
                        except Exception as e:
                            st.error(f"Failed to load index: {e}")
                            docs = []
                        if docs:
                            with st.spinner("Regenerating (plain text)..."):
                                prompt_template = build_plain_prompt()
                                google_api_key = st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
                                answer_text, model_used, error = generate_answer(prompt_template, docs, user_q, google_api_key=google_api_key)
                            if answer_text:
                                formatted_answer = html_module.escape(answer_text)
                                add_message('assistant', formatted_answer)
                                st.session_state.last_model_used = model_used
                                st.session_state.focus_index = len(st.session_state.history) - 1
                                st.success("Regenerated (plain text).")
                            else:
                                st.error(f"Regeneration failed: {error}")
            with cols_regen[1]:
                if st.button("Regenerate — Bullets", key=f"regen_bullets_{target_idx}"):
                    user_q = find_preceding_user_message_text(target_idx)
                    if not user_q:
                        st.error("Could not find the original user question.")
                    else:
                        try:
                            embeddings = LocalEmbeddings()
                            new_db = LocalHNSW.load_local(HNSW_DIR, embedding=embeddings)
                            docs = new_db.similarity_search(user_q, k=RETRIEVE_K, embedding=embeddings)
                        except Exception as e:
                            st.error(f"Failed to load index: {e}")
                            docs = []
                        if docs:
                            with st.spinner("Regenerating (bullets)..."):
                                prompt_template = build_bullets_prompt()
                                google_api_key = st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
                                answer_text, model_used, error = generate_answer(prompt_template, docs, user_q, google_api_key=google_api_key)
                            if answer_text:
                                formatted_answer = format_to_bullets(answer_text)
                                add_message('assistant', formatted_answer)
                                st.session_state.last_model_used = model_used
                                st.session_state.focus_index = len(st.session_state.history) - 1
                                st.success("Regenerated (bullets).")
                            else:
                                st.error(f"Regeneration failed: {error}")
        else:
            st.info("No assistant message available to regenerate.")

        st.markdown("---")
        st.subheader("File / Index")
        if st.button("Delete Index"):
            try:
                if os.path.isdir(HNSW_DIR):
                    shutil.rmtree(HNSW_DIR)
                st.session_state.hnsw_ready = False
                st.success("Index deleted.")
            except Exception as e:
                st.error(f"Failed to delete index: {e}")

        st.markdown("---")
        if st.session_state.last_model_used:
            st.write(f"_Last model used: {st.session_state.last_model_used}_")

if __name__ == "__main__":
    main()
