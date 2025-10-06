# chatpdf_local_embeddings.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
# removed: from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# from langchain.vectorstores import FAISS
#from langchain_community.vectorstores import FAISS

# Use HNSWLib vectorstore (more cloud-friendly than FAISS)
from langchain.vectorstores import HNSWLib

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted, NotFound
import time
import datetime
import shutil
import streamlit.components.v1 as components
import html as html_module
import re

# NEW: open-source embedding model (sentence-transformers)
from sentence_transformers import SentenceTransformer

# ---------------------------
# Local embedding wrapper for LangChain/FAISS compatibility
# ---------------------------
class LocalEmbeddings:
    """
    Minimal wrapper exposing embed_documents and embed_query so it can be
    passed into FAISS.from_texts and FAISS.load_local as the 'embeddings' object.
    Uses sentence-transformers under the hood.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # returns a list of list[float]
        vectors = self.model.encode(texts, show_progress_bar=False)
        return [vector.tolist() if hasattr(vector, "tolist") else list(map(float, vector)) for vector in vectors]

    def embed_query(self, text):
        vector = self.model.encode([text], show_progress_bar=False)[0]
        return vector.tolist() if hasattr(vector, "tolist") else list(map(float, vector))

# ---------------------------
# NOTE: If you previously had EMBEDDING_MODEL config for Gemini, you can ignore it now.
# ---------------------------

FAISS_DIR = "faiss_index"

# ---------------------------
# Helpers: PDF -> Text -> Chunks -> FAISS
# ---------------------------

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception:
            # skip unreadable files
            continue
    return text


def get_text_chunks(text):
    """Split long text into smaller chunks for embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300
    )
    return text_splitter.split_text(text)


HNSW_DIR = "hnsw_index"  # new directory for HNSW files

def get_vector_store(text_chunks, progress_callback=None):
    """
    Build embeddings using LocalEmbeddings (or your chosen embeddings object)
    and create/save an HNSWLib index.
    """
    embeddings = LocalEmbeddings()  # or GoogleGenerativeAIEmbeddings if you revert
    if progress_callback:
        progress_callback("Embedding texts and building HNSW index (local model)...")

    # HNSWLib.from_texts expects an 'embeddings' object with embed_documents method
    index = HNSWLib.from_texts(texts=text_chunks, embedding=embeddings, space='cosine')
    index.save_local(HNSW_DIR)

    if progress_callback:
        progress_callback("Saved HNSW index to disk.")

# ---------------------------
# Prompt builders (Plain / Bullets)
# ---------------------------

def build_plain_prompt():
    """Simple plain text prompt (concise answer then short explanation)."""
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
    """Prompt asking for bullets-style output."""
    template = (
        """
You are an assistant that answers using ONLY the provided context.

Start with a single-line direct answer (or "Answer is not available in the context."). Then include a 'Key points:' section with 3-6 bullet points listing the important facts or steps.

Prefer short bullet lines and avoid long paragraphs.

Context:
{context}

Question:
{question}
"""
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ---------------------------
# Preferred model order for chat (Gemini 2.0 if available; unchanged)
# ---------------------------
MODEL_ORDER = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-pro"
]

# ---------------------------
# QA + Model fallback (generalized to accept a prompt_template)
# ---------------------------

def generate_answer_with_fallback_using_prompt(prompt_template: PromptTemplate, docs, question):
    """
    Try models in MODEL_ORDER using the provided prompt_template.
    Returns: (response_text, model_name, error_or_none)
    """
    for model_name in MODEL_ORDER:
        try:
            model = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)

            response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

            text = None
            # Normalize common LangChain response shapes
            if isinstance(response, dict):
                for key in ("output_text", "text", "answer", "output"):
                    if key in response and response[key]:
                        text = response[key]
                        break
                if not text:
                    # scan values
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
                return text, model_name, None
            else:
                st.warning(f"Model {model_name} returned an empty response; trying next fallback...")
                continue

        except ResourceExhausted:
            st.warning(f"Quota exhausted for {model_name}. Trying next fallback...")
            continue
        except NotFound:
            st.warning(f"Model {model_name} not found. Trying next fallback...")
            continue
        except Exception as e:
            st.warning(f"Model {model_name} failed with: {str(e)}. Trying next fallback...")
            continue

    return None, None, "All models failed or exhausted their quotas."

# ---------------------------
# Session state initialization
# ---------------------------

def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "faiss_ready" not in st.session_state:
        st.session_state.faiss_ready = os.path.isdir(FAISS_DIR)
    if "last_model_used" not in st.session_state:
        st.session_state.last_model_used = None
    # Persistent focus index for highlighting a message
    if "focus_index" not in st.session_state:
        st.session_state.focus_index = None

# ---------------------------
# Timestamp formatting helper
# ---------------------------

def format_time(iso_ts: str) -> str:
    """
    Convert an ISO timestamp (or other common timestamp string) into
    a human-friendly 'YYYY-MM-DD HH:MM:SS' string for display.
    If parsing fails, return the original string.
    """
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
# Rendering: convert markdown-like assistant answers to HTML
# ---------------------------

def render_markdown_like_to_html(text: str) -> str:
    """
    Lightweight renderer that converts markdown-like structure into HTML:
    - handles fenced code blocks ```lang ... ```
    - converts lines starting with - or * into <ul>
    - converts numbered lists (1. 2.) into <ol>
    - converts headings (#, ##, ###)
    - preserves paragraphs and inline escaping
    """
    if not text:
        return ""

    escaped = html_module.escape(text)
    lines = escaped.splitlines()
    html_parts = []
    i = 0
    in_ul = False
    in_ol = False
    in_code = False
    code_lang = ""
    code_lines = []

    while i < len(lines):
        line = lines[i]

        # Fenced code block start/end (```lang)
        if not in_code and line.strip().startswith("```"):
            in_code = True
            code_lang = line.strip()[3:].strip()
            code_lines = []
            i += 1
            continue
        if in_code:
            if line.strip().startswith("```"):
                code_html = "<pre><code"
                if code_lang:
                    code_html += f" class='lang-{html_module.escape(code_lang)}'"
                code_html += ">" + "\n".join(code_lines) + "</code></pre>"
                html_parts.append(code_html)
                in_code = False
                code_lang = ""
                code_lines = []
                i += 1
                continue
            else:
                code_lines.append(line)
                i += 1
                continue

        # Headings
        if re.match(r"^#{1,6}\s+", line):
            level = len(re.match(r"^(#+)", line).group(1))
            content = line[level+1:].strip()
            html_parts.append(f"<h{level}>{content}</h{level}>")
            i += 1
            continue

        # Unordered list item
        if re.match(r"^\s*([-*])\s+", line):
            if not in_ul:
                in_ul = True
                html_parts.append("<ul>")
            content = re.sub(r"^\s*([-*])\s+", "", line)
            html_parts.append(f"<li>{content}</li>")
            i += 1
            # close ul when next line is not list item
            if i < len(lines):
                if not re.match(r"^\s*([-*])\s+", lines[i]):
                    html_parts.append("</ul>")
                    in_ul = False
            else:
                html_parts.append("</ul>")
                in_ul = False
            continue

        # Ordered list item
        if re.match(r"^\s*\d+\.\s+", line):
            if not in_ol:
                in_ol = True
                html_parts.append("<ol>")
            content = re.sub(r"^\s*\d+\.\s+", "", line)
            html_parts.append(f"<li>{content}</li>")
            i += 1
            if i < len(lines):
                if not re.match(r"^\s*\d+\.\s+", lines[i]):
                    html_parts.append("</ol>")
                    in_ol = False
            else:
                html_parts.append("</ol>")
                in_ol = False
            continue

        # Blank line -> paragraph separator
        if line.strip() == "":
            html_parts.append("<br/>")
            i += 1
            continue

        # Default: paragraph
        para_lines = [line]
        j = i + 1
        while j < len(lines) and lines[j].strip() != "" and not re.match(r"^\s*([-*])\s+", lines[j]) and not re.match(r"^\s*\d+\.\s+", lines[j]) and not re.match(r"^#{1,6}\s+", lines[j]) and not lines[j].strip().startswith("```"):
            para_lines.append(lines[j])
            j += 1
        paragraph = " ".join([l.strip() for l in para_lines])
        html_parts.append(f"<p>{paragraph}</p>")
        i = j

    if in_ul:
        html_parts.append("</ul>")
    if in_ol:
        html_parts.append("</ol>")
    return "\n".join(html_parts)

# ---------------------------
# UI helpers
# ---------------------------

def add_message(role, text):
    st.session_state.history.append({
        "role": role,
        "text": text,
        "time": datetime.datetime.now().isoformat()
    })

def find_preceding_user_message_text(idx):
    """
    Given an index in history (e.g., an assistant message index),
    find the nearest previous user message text. Return None if not found.
    """
    for i in range(idx - 1, -1, -1):
        if st.session_state.history[i]["role"] == "user":
            return st.session_state.history[i]["text"]
    return None

# ---------------------------
# Main app
# ---------------------------

def main():
    st.set_page_config(page_title="ChatPDF — Local Embeddings", layout="wide")
    init_session_state()

    # Header
    cols = st.columns([0.75, 0.25])
    with cols[0]:
        st.title("📄 ChatPDF — Local Embeddings (sentence-transformers)")
        st.caption("Upload PDFs, process them and chat — choose Plain or Bullets formatting.")
    with cols[1]:
        st.metric(label="", value="" if st.session_state.faiss_ready else " ")
        
    # Layout: Left = Conversations panel, Center = Chat, Right = Index & Controls
    left_col, center_col, right_col = st.columns([1.5, 3, 1])

    # --------------------
    # Left: Conversations Panel (auto-focus on radio selection)
    # --------------------
    with left_col:
        st.header("💬 Conversations")
        if not st.session_state.history:
            st.info("No messages yet — your conversation history will appear here.")
        else:
            preview_items = []
            for i, m in enumerate(st.session_state.history):
                role = "You" if m['role'] == 'user' else "Assistant"
                preview = m['text'][:80].replace("\n", " ")
                ts = format_time(m.get('time', ''))
                label = f"{i}: [{ts}] {role}: {preview}"
                preview_items.append(label)

            # Determine default index for radio (persist current focus if valid)
            default_index = 0
            if st.session_state.get("focus_index") is not None:
                fi = st.session_state.focus_index
                if 0 <= fi < len(preview_items):
                    default_index = fi

            # When user selects an item, immediately update focus_index (causes rerun)
            selected = st.radio(
                "Select message to focus",
                options=list(range(len(preview_items))),
                index=default_index,
                format_func=lambda x: preview_items[x]
            )
            if st.session_state.get("focus_index") != selected:
                st.session_state.focus_index = selected

            # Clear history + Export
            cols_left_actions = st.columns([0.5, 0.5])
            with cols_left_actions[0]:
                if st.button("Clear History"):
                    st.session_state.history = []
                    st.session_state.focus_index = None
                    st.success("Chat history cleared.")
            with cols_left_actions[1]:
                # placeholder to keep layout aligned
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

    # --------------------
    # Center: Input + Chat rendering (form first, then expander outside form)
    # --------------------
    with center_col:
        # Input area (form)
        with st.form(key="question_form", clear_on_submit=True):
            user_question = st.text_area("", placeholder="Type your question and press Send (Shift+Enter for newline)...", height=120, key='chat_input')
            format_choice = st.selectbox("Answer style", options=["Plain text", "Bullets"], index=0)
            cols_btn = st.columns([0.15, 0.85])
            with cols_btn[0]:
                send_btn = st.form_submit_button("Send")
            with cols_btn[1]:
                st.write(" ")

            if send_btn and user_question and user_question.strip():
                if not st.session_state.faiss_ready:
                    st.error("FAISS index not found. Please upload and process PDFs first (use Upload PDFs below).")
                else:
                    # Add user message then generate based on chosen style
                    add_message('user', user_question)
                    try:
                        embeddings = LocalEmbeddings()
                        new_db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
                        docs = new_db.similarity_search(user_question, k=4)
                    except Exception as e:
                        st.error(f"Failed to load FAISS index: {str(e)}")
                        docs = []

                    if docs:
                        with st.spinner("Generating answer (trying best available model)..."):
                            if format_choice == "Plain text":
                                prompt_template = build_plain_prompt()
                            else:
                                prompt_template = build_bullets_prompt()

                            answer_text, model_used, error = generate_answer_with_fallback_using_prompt(prompt_template, docs, user_question)

                        if answer_text:
                            # Append assistant message and focus it so it shows/highlights
                            add_message('assistant', answer_text)
                            st.session_state.last_model_used = model_used
                            st.session_state.focus_index = len(st.session_state.history) - 1
                            st.success("Answer generated and appended to conversation.")
                        else:
                            st.error(f"Failed to generate answer: {error}")

        # UPLOAD EXPANDER MOVED OUTSIDE THE FORM (fixes StreamlitAPIException)
        with st.expander("📎 Upload PDFs (attach & process here)"):
            uploaded = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])
            if st.button("Submit & Process Files"):
                if not uploaded:
                    st.warning("Please upload one or more PDF files first.")
                else:
                    progress_text = st.empty()
                    progress_bar = st.progress(0)

                    progress_text.info("Stage 1/3 — Extracting text from PDFs...")
                    time.sleep(0.2)
                    raw_text = get_pdf_text(uploaded)
                    progress_bar.progress(10)

                    if not raw_text.strip():
                        st.error("No readable text found in the uploaded PDFs.")
                    else:
                        progress_text.info("Stage 2/3 — Chunking text for embeddings...")
                        text_chunks = get_text_chunks(raw_text)
                        progress_bar.progress(40)

                        progress_text.info("Stage 3/3 — Creating embeddings and saving FAISS index...")
                        def cb(msg):
                            progress_text.info(msg)

                        get_vector_store(text_chunks, progress_callback=cb)
                        progress_bar.progress(100)

                        st.session_state.faiss_ready = True
                        st.success("✅ Processing complete — FAISS index saved.")

        # After form processing, render chat window so newly appended assistant appears immediately
        chat_box = st.container()
        st.markdown(
            """
            <style>
            .chat-window { max-height: 65vh; overflow: auto; padding: 12px; background: #ffffff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
            .message { margin: 8px 0; display: flex; align-items: flex-end; }
            .bubble { max-width: 78%; padding: 12px 14px; border-radius: 12px; line-height: 1.4; }
            .bubble.user { background: linear-gradient(90deg,#2b90ff,#1e6fe8); color: white; border-bottom-right-radius: 6px; margin-left: auto; }
            .bubble.assistant { background: #f1f3f5; color: #111827; border-bottom-left-radius: 6px; margin-right: auto; }
            .meta { font-size: 11px; color: #6b7280; margin-top: 4px; }
            .focused { box-shadow: 0 0 0 3px rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.4); border-radius: 12px; padding: 10px; background: linear-gradient(90deg, #fffefc, #f7fbff); }
            .bubble ul { margin: 8px 0 8px 18px; }
            .bubble ol { margin: 8px 0 8px 18px; }
            pre { background: #0b1220; color: #e6edf3; padding: 10px; border-radius: 8px; overflow: auto; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Build chat HTML from session_state.history (this reflects any additions made above)
        chat_html = "<div class='chat-window' id='chat-window'>"
        if not st.session_state.history:
            chat_html += "<div style='padding:20px;color:#6b7280'>No messages yet — upload PDFs and ask a question!</div>"
        else:
            for idx, msg in enumerate(st.session_state.history):
                ts = format_time(msg.get('time',''))
                msg_id = f"msg-{idx}"
                focused_class = "focused" if st.session_state.focus_index is not None and st.session_state.focus_index == idx else ""

                if msg['role'] == 'assistant':
                    content_html = render_markdown_like_to_html(msg['text'])
                    chat_html += (
                        f"<div class='message' id='{msg_id}'>"
                        f"<div class='bubble assistant {focused_class}'>{content_html}<div class='meta'>{ts}</div></div>"
                        f"</div>"
                    )
                else:
                    content_html = "<p>" + html_module.escape(msg['text']).replace("\n","<br/>") + "</p>"
                    chat_html += (
                        f"<div class='message' id='{msg_id}'>"
                        f"<div class='bubble user {focused_class}'>{content_html}<div class='meta'>{ts}</div></div>"
                        f"</div>"
                    )
        chat_html += "</div>"
        chat_box.markdown(chat_html, unsafe_allow_html=True)

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
                components.html(scroll_script, height=0, width=0)

    # --------------------
    # Right: Index Management & Regenerate Controls
    # --------------------
    with right_col:
        st.header("Controls")
        st.markdown("**FAISS status:**")
        if st.session_state.faiss_ready:
            st.success("FAISS index available.")
        else:
            st.warning("No FAISS index found.")

        st.markdown("---")
        st.subheader("Reformat Answer")
        # Determine target assistant message to regenerate (prefer focused assistant message)
        focus_idx = st.session_state.focus_index
        target_idx = None
        if focus_idx is not None and 0 <= focus_idx < len(st.session_state.history):
            if st.session_state.history[focus_idx]["role"] == "assistant":
                target_idx = focus_idx
            else:
                # try next assistant after user
                for j in range(focus_idx+1, len(st.session_state.history)):
                    if st.session_state.history[j]["role"] == "assistant":
                        target_idx = j
                        break
        else:
            # fallback to last assistant
            for j in range(len(st.session_state.history)-1, -1, -1):
                if st.session_state.history[j]["role"] == "assistant":
                    target_idx = j
                    break

        if target_idx is not None:
            st.write(f"Selected assistant message index: {target_idx}")
            st.write("Regenerate this answer in a new format (adds a new assistant message).")
            cols_regen = st.columns([1,1])
            with cols_regen[0]:
                if st.button("Regenerate — Plain Text", key=f"regen_plain_{target_idx}"):
                    user_q = find_preceding_user_message_text(target_idx)
                    if not user_q:
                        st.error("Could not locate the original user question to regenerate.")
                    else:
                        try:
                            embeddings = LocalEmbeddings()  # same wrapper you used to create the index
                            new_db = HNSWLib.load_local(HNSW_DIR, embeddings)
                            docs = new_db.similarity_search(user_question, k=4)
                        except Exception as e:
                            st.error(f"Failed to load FAISS index: {e}")
                            docs = []

                        if docs:
                            with st.spinner("Regenerating (plain text)..."):
                                prompt_template = build_plain_prompt()
                                answer_text, model_used, error = generate_answer_with_fallback_using_prompt(prompt_template, docs, user_q)
                            if answer_text:
                                add_message('assistant', answer_text)
                                st.session_state.last_model_used = model_used
                                st.session_state.focus_index = len(st.session_state.history) - 1
                                st.success("Regenerated (plain text)")
                            else:
                                st.error(f"Regeneration failed: {error}")
            with cols_regen[1]:
                if st.button("Regenerate — Bullets", key=f"regen_bullets_{target_idx}"):
                    user_q = find_preceding_user_message_text(target_idx)
                    if not user_q:
                        st.error("Could not locate the original user question to regenerate.")
                    else:
                        try:
                            embeddings = LocalEmbeddings()
                            new_db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
                            docs = new_db.similarity_search(user_q, k=4)
                        except Exception as e:
                            st.error(f"Failed to load FAISS index: {e}")
                            docs = []

                        if docs:
                            with st.spinner("Regenerating (bullets)..."):
                                prompt_template = build_bullets_prompt()
                                answer_text, model_used, error = generate_answer_with_fallback_using_prompt(prompt_template, docs, user_q)
                            if answer_text:
                                add_message('assistant', answer_text)
                                st.session_state.last_model_used = model_used
                                st.session_state.focus_index = len(st.session_state.history) - 1
                                st.success("Regenerated (bullets).")
                            else:
                                st.error(f"Regeneration failed: {error}")
        else:
            st.info("No assistant message available to regenerate. Send a question first.")

        st.markdown("---")
        st.subheader("File Uploads")
        if st.button("Delete All"):
            try:
                if os.path.isdir(FAISS_DIR):
                    shutil.rmtree(FAISS_DIR)
                st.session_state.faiss_ready = False
                st.success("FAISS index deleted.")
            except Exception as e:
                st.error(f"Failed to delete FAISS index: {e}")

        st.markdown("---")
        if st.session_state.last_model_used:
            st.write(f"_Last model used: {st.session_state.last_model_used}_")

if __name__ == '__main__':
    main()


