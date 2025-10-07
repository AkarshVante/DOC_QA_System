import os
import time
import shutil
import pickle
import html
import re
import numpy as np
import streamlit as st

# ---------- Optional heavy imports (handle if missing) ----------
try:
    from PyPDF2 import PdfReader
    HAVE_PYPDF2 = True
except ImportError:
    HAVE_PYPDF2 = False

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

try:
    import hnswlib
    HAVE_HNSWLIB = True
except ImportError:
    HAVE_HNSWLIB = False

try:
    from langchain.schema import Document
    from langchain.prompts import PromptTemplate
    from langchain.chains.question_answering import load_qa_chain
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from transformers import pipeline
    HF_FALLBACK_MODEL = "google/flan-t5-small"
    HAVE_TRANSFORMERS = True
except ImportError:
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
    "gemini-1.0-pro",
    "gemini-1.5-flash",
]

# ---------- Local Embeddings & HNSW ----------
class LocalEmbeddings:
    """Wrapper for sentence-transformers models."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        if not HAVE_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers is not installed. Please run 'pip install sentence-transformers'.")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """Embed a list of documents."""
        vectors = self.model.encode(texts, show_progress_bar=False)
        return [vec.tolist() for vec in vectors]

    def embed_query(self, text):
        """Embed a single query."""
        vec = self.model.encode([text], show_progress_bar=False)[0]
        return vec.tolist()


class LocalHNSW:
    """Wrapper for HNSWLib for local vector storage and search."""
    INDEX_FILENAME = "hnsw_index.bin"
    META_FILENAME = "hnsw_meta.pkl"

    def __init__(self, dim: int, space: str = "cosine"):
        if not HAVE_HNSWLIB:
            raise RuntimeError("hnswlib is not installed. Please run 'pip install hnswlib'.")
        self.dim = dim
        self.space = space
        self.index = hnswlib.Index(space=space, dim=dim)
        self._is_initialized = False
        self.id2doc = {}

    @classmethod
    def from_texts(cls, texts, embedding: LocalEmbeddings, space: str = "cosine"):
        """Create an HNSW index from a list of texts."""
        vectors = embedding.embed_documents(texts)
        arr = np.array(vectors, dtype=np.float32)
        dim = arr.shape[1]
        obj = cls(dim=dim, space=space)
        obj.index.init_index(max_elements=len(texts), ef_construction=200, M=16)
        ids = np.arange(len(texts), dtype=np.int64)
        obj.index.add_items(arr, ids)
        obj.index.set_ef(50)
        obj._is_initialized = True
        for i, t in enumerate(texts):
            obj.id2doc[int(i)] = Document(page_content=t)
        return obj

    def save_local(self, folder):
        """Save the index and metadata to a local folder."""
        os.makedirs(folder, exist_ok=True)
        idx_path = os.path.join(folder, self.INDEX_FILENAME)
        meta_path = os.path.join(folder, self.META_FILENAME)
        self.index.save_index(idx_path)
        with open(meta_path, "wb") as f:
            pickle.dump({"dim": self.dim, "space": self.space, "id2doc": self.id2doc}, f)

    @classmethod
    def load_local(cls, folder):
        """Load the index and metadata from a local folder."""
        meta_path = os.path.join(folder, cls.META_FILENAME)
        idx_path = os.path.join(folder, cls.INDEX_FILENAME)
        if not os.path.exists(meta_path) or not os.path.exists(idx_path):
            raise FileNotFoundError("HNSW index files not found in the specified folder.")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        obj = cls(dim=meta["dim"], space=meta["space"])
        obj.index.load_index(idx_path, max_elements=len(meta["id2doc"]))
        obj.index.set_ef(50)
        obj._is_initialized = True
        obj.id2doc = meta["id2doc"]
        return obj

    def similarity_search(self, query, k=4, embedding: LocalEmbeddings = None):
        """Perform a similarity search."""
        if not self._is_initialized:
            return []
        if embedding is None:
            raise ValueError("An embedding object with 'embed_query' method is required.")
        qvec = embedding.embed_query(query)
        qarr = np.array([qvec], dtype=np.float32)
        labels, _ = self.index.knn_query(qarr, k=k)
        return [self.id2doc[int(i)] for i in labels[0]]

# ---------- PDF Processing & Text Chunking ----------
def get_pdf_text(pdf_files):
    """Extract text from a list of PDF files."""
    text = ""
    if not HAVE_PYPDF2:
        st.error("PyPDF2 is not installed. Cannot process PDFs.")
        return ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.warning(f"⚠️ Couldn't read '{getattr(pdf, 'name', 'a file')}': {e}")
    return text

def get_text_chunks(text):
    """Split text into overlapping chunks."""
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_words = words[i:i + CHUNK_SIZE]
        chunks.append(" ".join(chunk_words))
    return chunks

# ---------- Answer Generation ----------
def build_prompt_template():
    """Builds a prompt template for the question-answering chain."""
    template = (
        "You are an expert assistant. Use the following pieces of context from PDF documents to answer the question at the end. "
        "Provide a concise and helpful answer based ONLY on the provided context. If you don't know the answer, just say that you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Helpful Answer:"
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])

def generate_answer(docs, question, google_api_key):
    """Generate an answer using the best available model (Gemini or local fallback)."""
    if not docs:
        return "⚠️ I couldn't find relevant information in the documents to answer your question. Please try rephrasing or check your PDFs.", None

    # Try Gemini via LangChain first
    if LANGCHAIN_AVAILABLE and google_api_key:
        for model_name in GEMINI_PREFERRED:
            try:
                llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=google_api_key, temperature=0.2)
                prompt_template = build_prompt_template()
                chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)
                response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
                return response.get("output_text", "No response text found.").strip(), model_name
            except Exception:
                continue # Try next preferred model

    # Fallback to local Transformers pipeline
    if HAVE_TRANSFORMERS:
        try:
            hf = pipeline("text2text-generation", model=HF_FALLBACK_MODEL, device=-1)
            context = "\n\n".join([d.page_content for d in docs])
            prompt_text = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
            resp = hf(prompt_text, max_length=300, num_return_sequences=1)
            return resp[0]['generated_text'].strip(), HF_FALLBACK_MODEL
        except Exception as e:
            return f"Generation failed with local model: {e}", None

    return "No generation model is available. Please set up a Google API key or install the 'transformers' library.", None

# ---------- Lexical Retriever Fallback ----------
def retrieve_top_chunks_lexical(query, chunks, top_k=RETRIEVE_K):
    """A simple keyword-based retriever as a fallback."""
    q_tokens = set(re.findall(r"\w+", query.lower()))
    if not q_tokens or not chunks:
        return []
    
    scored_chunks = []
    for chunk in chunks:
        t_tokens = set(re.findall(r"\w+", chunk.lower()))
        overlap = len(q_tokens.intersection(t_tokens))
        if overlap > 0:
            scored_chunks.append((overlap, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_docs = [Document(page_content=chunk) for _, chunk in scored_chunks[:top_k]]
    return top_docs

# ---------- Session State Initialization ----------
def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = os.path.isdir(HNSW_DIR)
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = []
    if "use_vector_search" not in st.session_state:
        st.session_state.use_vector_search = False

# ---------- UI & Styling ----------
UI_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}
body {
    background-color: #07101a;
}
/* Ensure sidebar is always visible and set its style */
[data-testid="stSidebar"] {
    background-color: #07101a;
    border-right: 1px solid #13303f;
}
/* Hide Streamlit's default elements */
#MainMenu, header, footer {
    visibility: hidden;
}
.stDeployButton {
    display: none;
}
/* Hide the sidebar collapse control */
button[data-testid="baseButton-header"] {
    display: none;
}
/* Custom styling for sidebar buttons */
[data-testid="stSidebar"] .stButton button {
    border-radius: 999px !important;
    border: 2px solid #add8e6 !important;
    background-color: transparent !important;
    color: #add8e6 !important;
    transition: all 0.2s ease-in-out;
    width: 100%;
}
[data-testid="stSidebar"] .stButton button:hover {
    background-color: rgba(173, 216, 230, 0.1) !important;
    color: #fff !important;
    border-color: #fff !important;
}
/* Status badge styling */
.status-badge {
    display: block; padding: 8px 16px; border-radius: 20px;
    font-size: 14px; font-weight: 600; margin: 12px 0; text-align: center;
}
.status-ready { background: rgba(25, 195, 125, 0.1); color: #19c37d; border: 1px solid rgba(25, 195, 125, 0.2); }
.status-not-ready { background: rgba(255, 102, 51, 0.1); color: #ff6633; border: 1px solid rgba(255, 102, 51, 0.2); }
/* Chat message styling */
.stChatMessage {
    background: #0b1520;
    border-radius: 18px;
    border: 1px solid #13303f;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
</style>
"""

# ---------- Main Application Logic ----------
def main():
    """Main function to run the Streamlit app."""
    st.markdown(UI_STYLES, unsafe_allow_html=True)
    init_session_state()
    google_api_key = st.secrets.get("GOOGLE_API_KEY")

    # --- Sidebar for document management ---
    with st.sidebar:
        st.markdown("## 📄 ChatPDF")
        st.markdown(
            "Upload PDFs and ask questions about their content. "
            "Your files are processed locally on this server."
        )

        badge_type = "ready" if st.session_state.vector_store_ready else "not-ready"
        badge_text = "Ready" if st.session_state.vector_store_ready else "No Documents"
        st.markdown(f'<div class="status-badge status-{badge_type}">Status: {badge_text}</div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # --- This is the PDF Uploader ---
        uploaded_files = st.file_uploader(
            "**1. Upload Your PDFs**",
            accept_multiple_files=True,
            type=['pdf'],
            help="Drag and drop one or more PDF files here."
        )

        if st.button("2. Process Documents", use_container_width=True, disabled=not uploaded_files):
            with st.spinner("Processing documents... This may take a moment."):
                try:
                    raw_text = get_pdf_text(uploaded_files)
                    if not raw_text.strip():
                        st.error("No readable text was found in the provided PDFs.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        st.session_state.text_chunks = text_chunks

                        if HAVE_SENTENCE_TRANSFORMERS and HAVE_HNSWLIB:
                            embeddings = LocalEmbeddings()
                            vector_store = LocalHNSW.from_texts(text_chunks, embedding=embeddings)
                            vector_store.save_local(HNSW_DIR)
                            st.session_state.use_vector_search = True
                            st.success(f"✅ Indexed {len(text_chunks)} chunks with vector search.")
                        else:
                            st.session_state.use_vector_search = False
                            st.info("Vector libraries not installed. Using keyword search fallback.")
                        
                        st.session_state.vector_store_ready = True
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
            st.rerun()

        st.markdown("---")
        st.markdown("**Manage Session**")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.session_state.vector_store_ready and st.button("Delete Documents", use_container_width=True):
            if os.path.isdir(HNSW_DIR):
                shutil.rmtree(HNSW_DIR)
            st.session_state.vector_store_ready = False
            st.session_state.use_vector_search = False
            st.session_state.text_chunks = []
            st.session_state.messages = []
            st.success("Documents and index deleted.")
            time.sleep(1)
            st.rerun()

    # --- Main Chat Interface ---
    st.markdown("<h1 style='text-align: center;'>Ask Your Documents</h1>", unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input and message handling
    prompt_placeholder = "Upload and process documents to begin..." if not st.session_state.vector_store_ready else "Ask a question..."
    prompt = st.chat_input(prompt_placeholder, disabled=not st.session_state.vector_store_ready)
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Searching and thinking..."):
                docs = []
                if st.session_state.use_vector_search:
                    try:
                        embeddings = LocalEmbeddings()
                        vector_store = LocalHNSW.load_local(HNSW_DIR)
                        docs = vector_store.similarity_search(prompt, k=RETRIEVE_K, embedding=embeddings)
                    except Exception:
                        st.warning("Vector search failed. Falling back to keyword search.")
                        docs = retrieve_top_chunks_lexical(prompt, st.session_state.text_chunks, top_k=RETRIEVE_K)
                else:
                    docs = retrieve_top_chunks_lexical(prompt, st.session_state.text_chunks, top_k=RETRIEVE_K)

                answer, model_used = generate_answer(docs, prompt, google_api_key)
                if model_used:
                    answer += f"\n\n*Powered by: {model_used}*"
                message_placeholder.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()


