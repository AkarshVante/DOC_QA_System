# import os
# import time
# import shutil
# import pickle
# import re
# import numpy as np
# import streamlit as st

# # ---------- Graceful Import of Heavy Libraries ----------
# # This pattern allows the app to run even if some optional libraries are not installed.

# try:
#     from PyPDF2 import PdfReader
#     HAVE_PYPDF2 = True
# except ImportError:
#     HAVE_PYPDF2 = False

# try:
#     from sentence_transformers import SentenceTransformer
#     HAVE_SENTENCE_TRANSFORMERS = True
# except ImportError:
#     HAVE_SENTENCE_TRANSFORMERS = False

# try:
#     import hnswlib
#     HAVE_HNSWLIB = True
# except ImportError:
#     HAVE_HNSWLIB = False

# try:
#     from langchain.schema import Document
#     from langchain.prompts import PromptTemplate
#     from langchain.chains.question_answering import load_qa_chain
#     from langchain_google_genai import ChatGoogleGenerativeAI
#     LANGCHAIN_AVAILABLE = True
# except ImportError:
#     LANGCHAIN_AVAILABLE = False

# try:
#     from transformers import pipeline
#     HF_FALLBACK_MODEL = "google/flan-t5-small"
#     HAVE_TRANSFORMERS = True
# except ImportError:
#     HAVE_TRANSFORMERS = False
#     HF_FALLBACK_MODEL = None

# # ---------- Application Configuration ----------
# st.set_page_config(page_title="ChatPDF", layout="wide", initial_sidebar_state="expanded")
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# HNSW_DIR = "hnsw_index"
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200
# RETRIEVE_K = 4
# # Prioritizing the requested Gemini models
# GEMINI_PREFERRED = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-pro"]

# # ---------- UI & Styling (Restored) ----------
# UI_STYLES = """
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
#     html, body, [class*="st-"] { 
#         font-family: 'Inter', sans-serif; 
#     }

#     /* Hide default Streamlit elements */
#     #MainMenu, footer, .stDeployButton { 
#         visibility: hidden; 
#     }
    
#     /* Main app background */
#     .stApp {
#         background-color: #07101a;
#     }

#     /* Style the sidebar */
#     [data-testid="stSidebar"] {
#         background-color: #07101a;
#         border-right: 1px solid #13303f;
#     }
    
#     /* Custom button style in sidebar */
#     [data-testid="stSidebar"] .stButton button {
#         border-radius: 999px;
#         border: 2px solid #add8e6;
#         background-color: transparent;
#         color: #add8e6;
#         transition: all 0.2s ease-in-out;
#     }
#     [data-testid="stSidebar"] .stButton button:hover {
#         background-color: rgba(173, 216, 230, 0.1);
#         color: #fff;
#         border-color: #fff;
#     }

#     /* Status badge styling */
#     .status-badge {
#         display: block; padding: 8px; border-radius: 20px;
#         font-weight: 600; margin: 12px auto; text-align: center;
#     }
#     .status-ready { background-color: rgba(25, 195, 125, 0.1); color: #19c37d; }
#     .status-not-ready { background-color: rgba(255, 102, 51, 0.1); color: #ff6633; }

# </style>
# """

# # ---------- Core Backend Classes ----------
# @st.cache_resource
# def get_embedding_model():
#     if not HAVE_SENTENCE_TRANSFORMERS:
#         st.error("Sentence Transformers library not found. Please install it.")
#         return None
#     return SentenceTransformer(EMBEDDING_MODEL_NAME)

# class LocalEmbeddings:
#     """Wrapper for sentence-transformers models."""
#     def __init__(self):
#         self.model = get_embedding_model()
#         if self.model is None:
#             raise RuntimeError("Sentence Transformer model could not be loaded.")

#     def embed_documents(self, texts):
#         return [vec.tolist() for vec in self.model.encode(texts, show_progress_bar=False)]

#     def embed_query(self, text):
#         return self.model.encode([text], show_progress_bar=False)[0].tolist()

# class LocalHNSW:
#     """Wrapper for HNSWLib, a local vector search index."""
#     INDEX_FILENAME = "hnsw_index.bin"
#     META_FILENAME = "hnsw_meta.pkl"

#     def __init__(self, dim: int, space: str = "cosine"):
#         if not HAVE_HNSWLIB:
#             raise RuntimeError("hnswlib is not installed.")
#         self.dim = dim
#         self.index = hnswlib.Index(space=space, dim=dim)
#         self.id2doc = {}

#     @classmethod
#     def from_texts(cls, texts, embedding: LocalEmbeddings):
#         vectors = np.array(embedding.embed_documents(texts), dtype=np.float32)
#         dim = vectors.shape[1]
#         obj = cls(dim=dim)
#         obj.index.init_index(max_elements=len(texts), ef_construction=200, M=16)
#         obj.index.add_items(vectors, np.arange(len(texts), dtype=np.int64))
#         obj.index.set_ef(50) # Set ef for search
#         obj.id2doc = {i: Document(page_content=t) for i, t in enumerate(texts)}
#         return obj

#     def save_local(self, folder):
#         os.makedirs(folder, exist_ok=True)
#         self.index.save_index(os.path.join(folder, self.INDEX_FILENAME))
#         with open(os.path.join(folder, self.META_FILENAME), "wb") as f:
#             pickle.dump({"dim": self.dim, "id2doc": self.id2doc}, f)

#     @classmethod
#     def load_local(cls, folder):
#         meta_path = os.path.join(folder, cls.META_FILENAME)
#         index_path = os.path.join(folder, cls.INDEX_FILENAME)
#         with open(meta_path, "rb") as f:
#             meta = pickle.load(f)
#         obj = cls(dim=meta["dim"])
#         num_elements = len(meta["id2doc"])
#         obj.index.load_index(index_path, max_elements=num_elements)
        
#         # *** BUG FIX IS HERE ***
#         # This crucial line prepares the loaded index for searching.
#         # It sets the 'ef' parameter, which controls search accuracy and performance.
#         # This resolves the "Cannot return the results in a contigious 2D array" error.
#         obj.index.set_ef(50) 
        
#         obj.id2doc = meta["id2doc"]
#         return obj

#     def similarity_search(self, query, k, embedding):
#         qvec = np.array([embedding.embed_query(query)], dtype=np.float32)
#         # Ensure k is not greater than the number of elements in the index
#         num_elements = self.index.get_current_count()
#         if k > num_elements:
#             k = num_elements
        
#         if k == 0:
#             return [] # Avoids error if index is empty
            
#         labels, _ = self.index.knn_query(qvec, k=k)
#         return [self.id2doc[int(i)] for i in labels[0]]

# # ---------- Document Processing and RAG Logic ----------
# def get_pdf_text(pdf_files):
#     if not HAVE_PYPDF2:
#         return ""
#     text = ""
#     for pdf in pdf_files:
#         try:
#             reader = PdfReader(pdf)
#             for page in reader.pages:
#                 if page_text := page.extract_text():
#                     text += page_text + "\n"
#         except Exception as e:
#             st.warning(f"Could not read '{getattr(pdf, 'name', 'file')}': {e}")
#     return text

# def get_text_chunks(text):
#     text = re.sub(r'\s+', ' ', text).strip()
#     words = text.split()
#     return [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP)]

# def build_prompt_template():
#     return PromptTemplate(
#         template="Use the following context to answer the question. Provide a concise answer based ONLY on the provided text. If the answer is not in the text, state that the information is not available in the document.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
#         input_variables=["context", "question"]
#     )

# def generate_answer(docs, question, google_api_key):
#     if not docs:
#         return "No relevant context was found in the documents for your question.", None
#     if LANGCHAIN_AVAILABLE and google_api_key:
#         for model_name in GEMINI_PREFERRED:
#             try:
#                 llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=google_api_key, temperature=0.1)
#                 chain = load_qa_chain(llm, chain_type="stuff", prompt=build_prompt_template())
#                 response = chain.invoke({"input_documents": docs, "question": question})
#                 return response.get("output_text", "Could not generate an answer.").strip(), model_name
#             except Exception:
#                 st.warning(f"Model '{model_name}' failed. Trying next...")
#                 continue
#     return "No generation model is available or all models failed.", None

# # ---------- Main Application Logic ----------
# def main():
#     st.markdown(UI_STYLES, unsafe_allow_html=True)

#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "vector_store_ready" not in st.session_state:
#         st.session_state.vector_store_ready = os.path.isdir(HNSW_DIR)

#     # --- Sidebar for Document and Session Management ---
#     with st.sidebar:
#         st.header("📄 ChatPDF")
#         st.markdown("Your personal document assistant.")
        
#         status_text = "Ready" if st.session_state.vector_store_ready else "No Documents"
#         status_class = "status-ready" if st.session_state.vector_store_ready else "status-not-ready"
#         st.markdown(f'<div class="status-badge {status_class}">Status: {status_text}</div>', unsafe_allow_html=True)
        
#         st.markdown("---")
#         st.subheader("1. Upload Documents")
#         uploaded_files = st.file_uploader(
#             "Upload your PDF files here.",
#             accept_multiple_files=True,
#             type=['pdf'],
#             label_visibility="collapsed"
#         )
        
#         if st.button("2. Process Documents", use_container_width=True, disabled=not uploaded_files):
#             if not all([HAVE_PYPDF2, HAVE_SENTENCE_TRANSFORMERS, HAVE_HNSWLIB]):
#                 st.error("A required library is missing. Please check installations.")
#             else:
#                 with st.spinner("Processing documents..."):
#                     raw_text = get_pdf_text(uploaded_files)
#                     if raw_text.strip():
#                         chunks = get_text_chunks(raw_text)
#                         embeddings = LocalEmbeddings()
#                         vector_store = LocalHNSW.from_texts(chunks, embeddings)
#                         vector_store.save_local(HNSW_DIR)
#                         st.session_state.vector_store_ready = True
#                         st.success("✅ Documents processed!")
#                         time.sleep(1)
#                         st.rerun()
#                     else:
#                         st.error("Processing failed. No readable text found in PDFs.")

#         st.markdown("---")
#         st.subheader("3. Manage Session")
#         if st.button("Clear Conversation", use_container_width=True):
#             st.session_state.messages = []
#             st.rerun()

#         if st.session_state.vector_store_ready and st.button("Delete Documents", use_container_width=True):
#             shutil.rmtree(HNSW_DIR, ignore_errors=True)
#             st.session_state.vector_store_ready = False
#             st.session_state.messages = []
#             st.success("Documents deleted.")
#             time.sleep(1)
#             st.rerun()

#     # --- Main Chat Interface ---
#     st.title("Ask Your Documents")

#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     prompt_placeholder = "Please process documents first..." if not st.session_state.vector_store_ready else "Ask a question..."
#     if prompt := st.chat_input(prompt_placeholder, disabled=not st.session_state.vector_store_ready):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 google_api_key = st.secrets.get("GOOGLE_API_KEY")
#                 embeddings = LocalEmbeddings()
#                 vector_store = LocalHNSW.load_local(HNSW_DIR)
#                 docs = vector_store.similarity_search(prompt, k=RETRIEVE_K, embedding=embeddings)
#                 answer, model = generate_answer(docs, prompt, google_api_key)
                
#                 if model:
#                     answer += f""
                
#                 st.markdown(answer)
#         st.session_state.messages.append({"role": "assistant", "content": answer})

# if __name__ == "__main__":
#     main()
import os
import time
import shutil
import pickle
import re
import numpy as np
import streamlit as st

# ---------- Graceful Import of Heavy Libraries ----------
# This pattern allows the app to run even if some optional libraries are not installed.

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

# ---------- Application Configuration ----------
st.set_page_config(page_title="ChatPDF", layout="wide", initial_sidebar_state="expanded")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HNSW_DIR = "hnsw_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVE_K = 4
# Prioritizing the requested Gemini models
GEMINI_PREFERRED = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-pro"]

# ---------- UI & Styling (Enhanced) ----------
UI_STYLES = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Define animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide other default Streamlit elements */
    #MainMenu, footer, .stDeployButton {
        visibility: hidden;
    }
    
    /* Main app background */
    .stApp {
        background-color: #07101a;
    }

    /* --- Sidebar Styling --- */
    [data-testid="stSidebar"] {
        background-color: #07101a;
        border-right: 1px solid #13303f;
    }
    
    /* Sidebar button with glowing effect */
    [data-testid="stSidebar"] .stButton button {
        border-radius: 999px;
        border: 1px solid #2c5970;
        background-color: transparent;
        color: #add8e6;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 0 5px 0px rgba(0, 150, 255, 0.3);
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: rgba(173, 216, 230, 0.1);
        color: #fff;
        border-color: #00aaff;
        box-shadow: 0 0 10px 2px rgba(0, 150, 255, 0.6);
    }
    
    /* Status badge styling */
    .status-badge {
        display: block; padding: 8px; border-radius: 20px;
        font-weight: 600; margin: 12px auto; text-align: center;
    }
    .status-ready { background-color: rgba(25, 195, 125, 0.1); color: #19c37d; }
    .status-not-ready { background-color: rgba(255, 102, 51, 0.1); color: #ff6633; }
    
    /* --- Main Chat Interface Styling --- */
    /* Chat message styling with animation and glow */
    [data-testid="stChatMessage"] {
        animation: fadeIn 0.5s ease-in-out;
        border-radius: 10px;
        border: 1px solid #13303f;
        background-color: #0a1929;
        box-shadow: 0 0 8px 1px rgba(0, 150, 255, 0.15);
        margin: 10px 0;
    }
    
    /* Increase size of the chat input box */
    [data-testid="stChatInput"] textarea {
        min-height: 10px;
        font-size: 1.1em;
         color: #FFFFFF;
    }
</style>
"""

# ---------- Core Backend Classes ----------
@st.cache_resource
def get_embedding_model():
    if not HAVE_SENTENCE_TRANSFORMERS:
        st.error("Sentence Transformers library not found. Please install it.")
        return None
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

class LocalEmbeddings:
    """Wrapper for sentence-transformers models."""
    def __init__(self):
        self.model = get_embedding_model()
        if self.model is None:
            raise RuntimeError("Sentence Transformer model could not be loaded.")

    def embed_documents(self, texts):
        return [vec.tolist() for vec in self.model.encode(texts, show_progress_bar=False)]

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

class LocalHNSW:
    """Wrapper for HNSWLib, a local vector search index."""
    INDEX_FILENAME = "hnsw_index.bin"
    META_FILENAME = "hnsw_meta.pkl"

    def __init__(self, dim: int, space: str = "cosine"):
        if not HAVE_HNSWLIB:
            raise RuntimeError("hnswlib is not installed.")
        self.dim = dim
        self.index = hnswlib.Index(space=space, dim=dim)
        self.id2doc = {}

    @classmethod
    def from_texts(cls, texts, embedding: LocalEmbeddings):
        vectors = np.array(embedding.embed_documents(texts), dtype=np.float32)
        dim = vectors.shape[1]
        obj = cls(dim=dim)
        obj.index.init_index(max_elements=len(texts), ef_construction=200, M=16)
        obj.index.add_items(vectors, np.arange(len(texts), dtype=np.int64))
        obj.index.set_ef(50) # Set ef for search
        obj.id2doc = {i: Document(page_content=t) for i, t in enumerate(texts)}
        return obj

    def save_local(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.index.save_index(os.path.join(folder, self.INDEX_FILENAME))
        with open(os.path.join(folder, self.META_FILENAME), "wb") as f:
            pickle.dump({"dim": self.dim, "id2doc": self.id2doc}, f)

    @classmethod
    def load_local(cls, folder):
        meta_path = os.path.join(folder, cls.META_FILENAME)
        index_path = os.path.join(folder, cls.INDEX_FILENAME)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        obj = cls(dim=meta["dim"])
        num_elements = len(meta["id2doc"])
        obj.index.load_index(index_path, max_elements=num_elements)
        
        obj.index.set_ef(50) 
        
        obj.id2doc = meta["id2doc"]
        return obj

    def similarity_search(self, query, k, embedding):
        qvec = np.array([embedding.embed_query(query)], dtype=np.float32)
        num_elements = self.index.get_current_count()
        if k > num_elements:
            k = num_elements
        
        if k == 0:
            return [] # Avoids error if index is empty
            
        labels, _ = self.index.knn_query(qvec, k=k)
        return [self.id2doc[int(i)] for i in labels[0]]

# ---------- Document Processing and RAG Logic ----------
def get_pdf_text(pdf_files):
    if not HAVE_PYPDF2:
        return ""
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                if page_text := page.extract_text():
                    text += page_text + "\n"
        except Exception as e:
            st.warning(f"Could not read '{getattr(pdf, 'name', 'file')}': {e}")
    return text

def get_text_chunks(text):
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    return [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP)]

def build_prompt_template():
    return PromptTemplate(
        template="Use the following context to answer the question. Provide a concise answer based ONLY on the provided text. If the answer is not in the text, state that the information is not available in the document.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        input_variables=["context", "question"]
    )

def generate_answer(docs, question, google_api_key):
    if not docs:
        return "No relevant context was found in the documents for your question.", None
    if LANGCHAIN_AVAILABLE and google_api_key:
        for model_name in GEMINI_PREFERRED:
            try:
                llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=google_api_key, temperature=0.1)
                chain = load_qa_chain(llm, chain_type="stuff", prompt=build_prompt_template())
                response = chain.invoke({"input_documents": docs, "question": question})
                return response.get("output_text", "Could not generate an answer.").strip(), model_name
            except Exception:
                st.warning(f"Model '{model_name}' failed. Trying next...")
                continue
    return "No generation model is available or all models failed.", None

# ---------- Main Application Logic ----------
def main():
    st.markdown(UI_STYLES, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = os.path.isdir(HNSW_DIR)

    # --- Sidebar for Document and Session Management ---
    with st.sidebar:
        st.header("📄 ChatPDF")
        st.markdown("Your personal document assistant.")
        
        status_text = "Ready" if st.session_state.vector_store_ready else "No Documents"
        status_class = "status-ready" if st.session_state.vector_store_ready else "status-not-ready"
        st.markdown(f'<div class="status-badge {status_class}">Status: {status_text}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("1. Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your PDF files here.",
            accept_multiple_files=True,
            type=['pdf'],
            label_visibility="collapsed"
        )
        
        if st.button("2. Process Documents", use_container_width=True, disabled=not uploaded_files):
            if not all([HAVE_PYPDF2, HAVE_SENTENCE_TRANSFORMERS, HAVE_HNSWLIB]):
                st.error("A required library is missing. Please check installations.")
            else:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(uploaded_files)
                    if raw_text.strip():
                        chunks = get_text_chunks(raw_text)
                        embeddings = LocalEmbeddings()
                        vector_store = LocalHNSW.from_texts(chunks, embeddings)
                        vector_store.save_local(HNSW_DIR)
                        st.session_state.vector_store_ready = True
                        st.success("✅ Documents processed!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Processing failed. No readable text found in PDFs.")

        st.markdown("---")
        st.subheader("3. Manage Session")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.session_state.vector_store_ready and st.button("Delete Documents", use_container_width=True):
            shutil.rmtree(HNSW_DIR, ignore_errors=True)
            st.session_state.vector_store_ready = False
            st.session_state.messages = []
            st.success("Documents deleted.")
            time.sleep(1)
            st.rerun()

    # --- Main Chat Interface (with Enhanced Title) ---
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="font-size: 3em; font-weight: 700; color: #FFFFFF;">
            <span style="margin-right: 15px;">📄</span>Chat With Your Documents
        </h1>
    </div>
    """, unsafe_allow_html=True)


    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt_placeholder = "Please process documents first..." if not st.session_state.vector_store_ready else "Ask a question..."
    if prompt := st.chat_input(prompt_placeholder, disabled=not st.session_state.vector_store_ready):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                google_api_key = st.secrets.get("GOOGLE_API_KEY")
                embeddings = LocalEmbeddings()
                vector_store = LocalHNSW.load_local(HNSW_DIR)
                docs = vector_store.similarity_search(prompt, k=RETRIEVE_K, embedding=embeddings)
                answer, model = generate_answer(docs, prompt, google_api_key)
                
                if model:
                    answer += f""
                
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()








