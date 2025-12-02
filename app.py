# """
# Complete `app.py` for the Streamlit ChatPDF app.
# This version includes robust imports and fallbacks for different LangChain installs,
# clear user-facing error messages when runtime components are missing, and the
# same UI/CSS you provided.

# NOTE: Make sure your virtualenv has compatible packages installed (see README
# or requirements.txt suggestions). Typical helpful packages: langchain, langchain-core,
# langchain-text-splitters, langchain-google-genai, langchain-community, google-generativeai,
# PyPDF2, streamlit, sentence-transformers, faiss-cpu, python-dotenv, tiktoken.
# """

# import os
# import time
# import json
# import shutil
# from dotenv import load_dotenv

# import streamlit as st
# from PyPDF2 import PdfReader
# import google.generativeai as genai

# # ------------------------- Robust imports -------------------------
# # Text splitter: prefer the separated package, fallback to older langchain path
# try:
#     from langchain_text_splitters import RecursiveCharacterTextSplitter
# except Exception:
#     try:
#         from langchain.text_splitter import RecursiveCharacterTextSplitter
#     except Exception as e:
#         RecursiveCharacterTextSplitter = None
#         _text_splitter_import_error = e

# # langchain-google-genai chat wrapper (Gemini models)
# try:
#     from langchain_google_genai import ChatGoogleGenerativeAI
#     CHAT_GOOGLE_AVAILABLE = True
# except Exception:
#     ChatGoogleGenerativeAI = None
#     CHAT_GOOGLE_AVAILABLE = False

# # langchain-community FAISS & embeddings (local sentence-transformer embeddings)
# try:
#     from langchain_community.vectorstores import FAISS
#     from langchain_huggingface import HuggingFaceEmbeddings
#     LCC_FAISS_AVAILABLE = True
# except Exception:
#     FAISS = None
#     SentenceTransformerEmbeddings = None
#     LCC_FAISS_AVAILABLE = False

# # Generic langchain imports used later (wrapped in try/except at use-time to avoid
# # import-time crashes when a package is missing)
# try:
#     from langchain.chains.question_answering import load_qa_chain
#     from langchain.prompts import PromptTemplate
# except Exception:
#     load_qa_chain = None
#     PromptTemplate = None

# # ------------------------- Configuration -------------------------
# load_dotenv()

# st.set_page_config(
#     page_title="ChatPDF",
#     page_icon="📄",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # --- CSS styling (kept from your original file) ---
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

#     @keyframes fadeIn {
#         from { opacity: 0; transform: translateY(10px); }
#         to { opacity: 1; transform: translateY(0); }
#     }

#     html, body, [class*="st-"] {
#         font-family: 'Inter', sans-serif;
#     }
#     #MainMenu, footer, .stDeployButton {
#         visibility: hidden;
#     }
#     .stApp { background-color: #07101a; }
#     h1, h2, h3 { color: #ffffff; }
#     .title-glow { font-size: 2.5em; color: #ffffff; text-align: center; text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 30px #00ff00; }
#     [data-testid="stSidebar"] { background-color: #07101a; border-right: 1px solid #13303f; }
#     [data-testid="stSidebar"] .stButton button, [data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
#         border-radius: 999px; border: 1px solid #2c5970; background-color: transparent; color: #add8e6; transition: all 0.2s ease-in-out; box-shadow: 0 0 5px 0px rgba(0, 150, 255, 0.3);
#     }
#     [data-testid="stSidebar"] .stButton button:hover, [data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
#         background-color: rgba(173, 216, 230, 0.1); color: #fff; border-color: #00aaff; box-shadow: 0 0 10px 2px rgba(0, 150, 255, 0.6);
#     }
#     .status-badge { display: block; padding: 8px; border-radius: 20px; font-weight: 600; margin: 12px auto; text-align: center; }
#     .status-ready { background-color: rgba(25, 195, 125, 0.1); color: #19c37d; }
#     .status-not-ready { background-color: rgba(255, 102, 51, 0.1); color: #ff6633; }
#     .stChatMessage { animation: fadeIn 0.5s ease-out; transition: all 0.2s ease-in-out; }
#     div[data-testid="stChatMessage"] div[data-testid^="stMarkdownContainer"] { border-radius: 12px; padding: 14px 18px; margin: 4px; color: white; border: 1px solid #2c5970; box-shadow: 0 0 8px 1px rgba(0, 150, 255, 0.15); }
#     div[data-testid="stChatMessage-assistant"] div[data-testid^="stMarkdownContainer"] { background-color: #262D31; border-bottom-left-radius: 4px; }
#     div[data-testid="stChatMessage-user"] div[data-testid^="stMarkdownContainer"] { background-color: transparent; border-bottom-right-radius: 4px; }
#     [data-testid="stChatInput"] textarea { color: #FFFFFF; max-height: 150px; }
#     .stButton>button { background-color: transparent !important; color: #add8e6 !important; border: 1px solid #2c5970 !important; padding: 2px 10px !important; font-size: 0.8rem !important; border-radius: 999px !important; margin: -8px 0 10px 10px; }
#     .stButton>button:hover { border-color: #00aaff !important; color: #fff !important; }
# </style>
# """, unsafe_allow_html=True)

# # ------------------------- Google API configuration -------------------------
# GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     st.error("Google API key not found. Please set it in Streamlit secrets or as an environment variable.")
#     st.stop()

# try:
#     genai.configure(api_key=GOOGLE_API_KEY)
# except Exception as e:
#     st.error(f"Failed to configure Google Generative AI: {e}")
#     st.stop()

# # ------------------------- Constants -------------------------
# MODEL_ORDER = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# FAISS_DIR = "faiss_index"
# CHAT_HISTORY_FILE = "chat_history.json"

# # ------------------------- Helper functions -------------------------

# def get_pdf_text(pdf_docs):
#     """Extract text from a list of uploaded PDF files."""
#     text = ""
#     for pdf in pdf_docs:
#         try:
#             pdf_reader = PdfReader(pdf)
#             for i, page in enumerate(pdf_reader.pages):
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += f"\n--- Source: {pdf.name}, Page: {i+1} ---\n{page_text}"
#         except Exception as e:
#             st.warning(f"Could not read file: {pdf.name}. Error: {e}")
#     return text


# def get_text_chunks(text):
#     """Split text into manageable chunks for processing."""
#     if RecursiveCharacterTextSplitter is None:
#         raise ImportError("Text splitter not available. Install 'langchain-text-splitters' or use a compatible langchain.")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks


# @st.cache_resource
# def get_embedding_model():
#     """Load the sentence transformer model, cached for performance."""
#     if SentenceTransformerEmbeddings is None:
#         raise ImportError("SentenceTransformerEmbeddings not available. Install 'langchain-community' or provide an alternative embedding model.")
#     return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# def get_vector_store(text_chunks):
#     """Create and save a FAISS vector store from text chunks using a local model."""
#     if FAISS is None or SentenceTransformerEmbeddings is None:
#         st.error("FAISS or embeddings are not available in this environment. Ensure 'langchain-community' and 'faiss-cpu' are installed.")
#         st.session_state.faiss_ready = False
#         return None
#     try:
#         embeddings = get_embedding_model()
#         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#         vector_store.save_local(FAISS_DIR)
#         st.session_state.faiss_ready = True
#         return vector_store
#     except Exception as e:
#         st.error(f"Error creating vector store: {e}")
#         st.session_state.faiss_ready = False
#         return None


# def get_conversational_chain(prompt_template):
#     """Create a conversational QA chain with a custom prompt and model fallback."""
#     if PromptTemplate is None or load_qa_chain is None:
#         st.error("LangChain QA utilities are not available. Install/upgrade langchain.")
#         return None

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#     # First try Google Gemini via langchain-google-genai (preferred for your config)
#     if CHAT_GOOGLE_AVAILABLE and ChatGoogleGenerativeAI is not None:
#         for model_name in MODEL_ORDER:
#             try:
#                 model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
#                 chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#                 st.session_state.last_model_used = model_name
#                 return chain
#             except Exception:
#                 # try the next model
#                 continue
#         st.warning("langchain-google-genai is available but no Gemini models succeeded. Falling back to other chat models if available.")

#     # Fallback: try OpenAI Chat if installed and configured (best-effort)
#     try:
#         from langchain.chat_models import ChatOpenAI
#         openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
#         if openai_key:
#             for candidate in ["gpt-4", "gpt-3.5-turbo"]:
#                 try:
#                     model = ChatOpenAI(model_name=candidate, temperature=0.3)
#                     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#                     st.session_state.last_model_used = candidate
#                     st.warning(f"Using OpenAI model {candidate} as fallback.")
#                     return chain
#                 except Exception:
#                     continue
#     except Exception:
#         # OpenAI chat not available in environment — that's fine, we'll show an error below.
#         pass

#     st.error("No chat model is available in this environment. Install/upgrade 'langchain-google-genai' + 'langchain-core' or configure OpenAI.")
#     return None


# def format_chat_history(messages):
#     """Formats the chat history for downloading."""
#     chat_str = "Chat History\n"
#     chat_str += "=" * 20 + "\n\n"
#     for msg in messages:
#         role = "User" if msg.get("role") == "user" else "Assistant"
#         content = msg.get('content', '')
#         chat_str += f"[{role}]:\n{content}\n\n"
#         chat_str += "-" * 20 + "\n\n"
#     return chat_str


# def save_chat_history():
#     with open(CHAT_HISTORY_FILE, "w") as f:
#         json.dump(st.session_state.messages, f)


# def load_chat_history():
#     if os.path.exists(CHAT_HISTORY_FILE):
#         with open(CHAT_HISTORY_FILE, "r") as f:
#             return json.load(f)
#     return []

# # ------------------------- Main application -------------------------

# def main():
#     # Init session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = load_chat_history()
#     if "faiss_ready" not in st.session_state:
#         st.session_state.faiss_ready = os.path.isdir(FAISS_DIR)
#     if "source_toggle" not in st.session_state:
#         st.session_state.source_toggle = {}

#     # Sidebar
#     with st.sidebar:
#         st.header("📄 ChatPDF")
#         st.markdown("Your personal document assistant.")

#         status_text = "Ready" if st.session_state.faiss_ready else "No Documents"
#         status_class = "status-ready" if st.session_state.faiss_ready else "status-not-ready"
#         st.markdown(f'<div class="status-badge {status_class}">Status: {status_text}</div>', unsafe_allow_html=True)

#         st.markdown("---")
#         st.subheader("1. Upload Documents")
#         uploaded_files = st.file_uploader(
#             "Upload your PDF files here.",
#             accept_multiple_files=True,
#             type=["pdf"],
#             label_visibility="collapsed",
#         )

#         process_disabled = not uploaded_files or not (FAISS and SentenceTransformerEmbeddings and RecursiveCharacterTextSplitter)
#         if process_disabled and uploaded_files:
#             st.info("Processing disabled because required packages are missing. See the instructions in the app header or check requirements.txt.")

#         if st.button("2. Process Documents", use_container_width=True, disabled=process_disabled):
#             with st.spinner("Processing documents... This may take a moment."):
#                 raw_text = get_pdf_text(uploaded_files)
#                 if raw_text.strip():
#                     try:
#                         chunks = get_text_chunks(raw_text)
#                         get_vector_store(chunks)
#                         st.success("✅ Documents processed!")
#                         time.sleep(1)
#                         st.rerun()
#                     except Exception as e:
#                         st.error(f"Processing failed: {e}")
#                 else:
#                     st.error("Processing failed. No readable text found in PDFs.")

#         st.markdown("---")
#         st.subheader("2. Advanced Options")
#         if st.button("📝 Summarize PDF", use_container_width=True, disabled=not st.session_state.faiss_ready):
#             with st.spinner("Summarizing document..."):
#                 try:
#                     embeddings = get_embedding_model()
#                     vector_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
#                     docs = vector_store.similarity_search("Summarize the entire document", k=min(10, len(vector_store.index_to_docstore_id)))

#                     summary_prompt_template = """
#                     Based on the following context, provide a concise summary in bullet points.
#                     Focus on the key topics, findings, and conclusions.\n\n
#                     Context:\n {context}\n
#                     Question: \n{question}\n
#                     Summary:
#                     """
#                     chain = get_conversational_chain(summary_prompt_template)
#                     if chain:
#                         response = chain({"input_documents": docs, "question": "Summarize the entire document."}, return_only_outputs=True)
#                         summary = response.get("output_text", "Could not generate a summary.")
#                         st.session_state.messages.append({"role": "assistant", "content": summary, "sources": [doc.page_content[:150] + "..." for doc in docs]})
#                         save_chat_history()
#                         st.rerun()

#                 except Exception as e:
#                     st.error(f"An error occurred during summarization: {e}")

#         st.markdown("---")
#         st.subheader("3. Manage Session")
#         if st.button("Clear Conversation", use_container_width=True):
#             st.session_state.messages = []
#             st.session_state.source_toggle = {}
#             save_chat_history()
#             st.rerun()

#         if st.session_state.messages:
#             chat_history_str = format_chat_history(st.session_state.messages)
#             st.download_button(
#                 label="Download Chat",
#                 data=chat_history_str,
#                 file_name=f"chat_history_{time.strftime('%Y%m%d_%H%M%S')}.txt",
#                 mime="text/plain",
#                 use_container_width=True,
#             )

#         if st.session_state.faiss_ready and st.button("Delete Documents", use_container_width=True):
#             shutil.rmtree(FAISS_DIR, ignore_errors=True)
#             st.session_state.faiss_ready = False
#             st.session_state.messages = []
#             st.session_state.source_toggle = {}
#             save_chat_history()
#             st.success("Documents and index deleted.")
#             time.sleep(1)
#             st.rerun()

#     # Main header
#     st.markdown(
#         """
#         <div style="text-align: center; margin-bottom: 20px;">
#             <h1 style="font-size: 3em; font-weight: 700; color: #FFFFFF;">
#                 <span style="margin-right: 15px;">📄</span>Chat With Your Documents
#             </h1>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     # Display chat messages
#     for idx, msg in enumerate(st.session_state.messages):
#         with st.chat_message(msg.get("role", "assistant")):
#             st.markdown(msg.get("content", ""))

#         if msg.get("role") == "assistant" and msg.get("sources"):
#             if st.button("📄 View Source", key=f"src_{idx}"):
#                 st.session_state.source_toggle[idx] = not st.session_state.source_toggle.get(idx, False)
#             if st.session_state.source_toggle.get(idx, False):
#                 st.info("".join(msg.get("sources")))

#     # Chat input
#     prompt_placeholder = "Please process documents first..." if not st.session_state.faiss_ready else "Ask a question..."
#     if prompt := st.chat_input(prompt_placeholder, disabled=not st.session_state.faiss_ready):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     embeddings = get_embedding_model()
#                     vector_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
#                     docs = vector_store.similarity_search(prompt, k=4)

#                     qa_prompt_template = """
#                     Answer the question as detailed as possible from the provided context. If the answer is not in
#                     the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.\n\n
#                     Context:\n {context}\n
#                     Question: \n{question}\n
#                     Answer:
#                     """
#                     chain = get_conversational_chain(qa_prompt_template)
#                     if chain:
#                         response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)
#                         answer = response.get("output_text", "Sorry, I couldn't generate a response.")
#                     else:
#                         answer = "The conversation chain could not be initialized. Please check the logs and dependencies."

#                     st.markdown(answer)
#                     sources = [doc.page_content[:150] + "..." for doc in docs]
#                     st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
#                     save_chat_history()

#                 except Exception as e:
#                     error_message = f"An error occurred: {e}"
#                     st.error(error_message)
#                     st.session_state.messages.append({"role": "assistant", "content": error_message})
#                     save_chat_history()
#         st.rerun()


# if __name__ == "__main__":
#     main()








import os
import time
import json
import shutil
from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader

# --- Modern LangChain Imports (2025 Standard) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# ------------------------- Configuration -------------------------
load_dotenv()

st.set_page_config(
    page_title="ChatPDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- CSS Styling (Preserved) -------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    #MainMenu, footer, .stDeployButton {
        visibility: hidden;
    }
    .stApp { background-color: #07101a; }
    h1, h2, h3 { color: #ffffff; }
    .title-glow { font-size: 2.5em; color: #ffffff; text-align: center; text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 30px #00ff00; }
    [data-testid="stSidebar"] { background-color: #07101a; border-right: 1px solid #13303f; }
    [data-testid="stSidebar"] .stButton button, [data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
        border-radius: 999px; border: 1px solid #2c5970; background-color: transparent; color: #add8e6; transition: all 0.2s ease-in-out; box-shadow: 0 0 5px 0px rgba(0, 150, 255, 0.3);
    }
    [data-testid="stSidebar"] .stButton button:hover, [data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
        background-color: rgba(173, 216, 230, 0.1); color: #fff; border-color: #00aaff; box-shadow: 0 0 10px 2px rgba(0, 150, 255, 0.6);
    }
    .status-badge { display: block; padding: 8px; border-radius: 20px; font-weight: 600; margin: 12px auto; text-align: center; }
    .status-ready { background-color: rgba(25, 195, 125, 0.1); color: #19c37d; }
    .status-not-ready { background-color: rgba(255, 102, 51, 0.1); color: #ff6633; }
    .stChatMessage { animation: fadeIn 0.5s ease-out; transition: all 0.2s ease-in-out; }
    div[data-testid="stChatMessage"] div[data-testid^="stMarkdownContainer"] { border-radius: 12px; padding: 14px 18px; margin: 4px; color: white; border: 1px solid #2c5970; box-shadow: 0 0 8px 1px rgba(0, 150, 255, 0.15); }
    div[data-testid="stChatMessage-assistant"] div[data-testid^="stMarkdownContainer"] { background-color: #262D31; border-bottom-left-radius: 4px; }
    div[data-testid="stChatMessage-user"] div[data-testid^="stMarkdownContainer"] { background-color: transparent; border-bottom-right-radius: 4px; }
    [data-testid="stChatInput"] textarea { color: #FFFFFF; max-height: 150px; }
    .stButton>button { background-color: transparent !important; color: #add8e6 !important; border: 1px solid #2c5970 !important; padding: 2px 10px !important; font-size: 0.8rem !important; border-radius: 999px !important; margin: -8px 0 10px 10px; }
    .stButton>button:hover { border-color: #00aaff !important; color: #fff !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------- Setup & Constants -------------------------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Check for API Key immediately
if not GOOGLE_API_KEY:
    st.error("🚨 Google API Key missing! Please set `GOOGLE_API_KEY` in your `.env` file or Streamlit secrets.")
    st.stop()

# Constants
MODEL_ORDER = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_DIR = "faiss_index"
CHAT_HISTORY_FILE = "chat_history.json"

# ------------------------- Core Functions -------------------------

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Source: {pdf.name}, Page: {i+1} ---\n{page_text}"
        except Exception as e:
            st.warning(f"Could not read {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    """Split text using the updated RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

@st.cache_resource
def get_embedding_model():
    """
    Initialize HuggingFace Embeddings.
    Updated to use `langchain_huggingface` to avoid deprecation warnings.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_vector_store(text_chunks):
    """Create and save FAISS index locally."""
    try:
        embeddings = get_embedding_model()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_DIR)
        st.session_state.faiss_ready = True
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

def get_qa_chain(prompt_template_str):
    """
    Creates a QA chain using LCEL (LangChain Expression Language).
    Replaces the deprecated `load_qa_chain`.
    """
    # 1. Define the Prompt
    prompt = ChatPromptTemplate.from_template(prompt_template_str)

    # 2. Initialize the Model (Try models in order)
    llm = None
    for model_name in MODEL_ORDER:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.3,
                google_api_key=GOOGLE_API_KEY
            )
            break
        except Exception:
            continue
            
    if not llm:
        st.error("Could not initialize any Gemini models. Check your API key or quota.")
        return None

    # 3. Create the Document Chain (modern "stuff" chain)
    # This chain takes {context, question} and returns the answer.
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    return document_chain

# ------------------------- Session Management -------------------------

def save_chat_history():
    try:
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump(st.session_state.messages, f)
    except Exception as e:
        print(f"Error saving chat: {e}")

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def format_chat_history(messages):
    chat_str = "Chat History\n" + "=" * 20 + "\n\n"
    for msg in messages:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get('content', '')
        chat_str += f"[{role}]:\n{content}\n\n" + "-" * 20 + "\n\n"
    return chat_str

# ------------------------- Main App -------------------------

def main():
    # Session State Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()
    if "faiss_ready" not in st.session_state:
        st.session_state.faiss_ready = os.path.isdir(FAISS_DIR)
    if "source_toggle" not in st.session_state:
        st.session_state.source_toggle = {}

    # --- Sidebar ---
    with st.sidebar:
        st.header("📄 ChatPDF")
        st.markdown("Your personal document assistant.")

        # Status Indicator
        status_text = "Ready" if st.session_state.faiss_ready else "No Documents"
        status_class = "status-ready" if st.session_state.faiss_ready else "status-not-ready"
        st.markdown(f'<div class="status-badge {status_class}">Status: {status_text}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("1. Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs", 
            accept_multiple_files=True, 
            type=["pdf"], 
            label_visibility="collapsed"
        )

        if st.button("2. Process Documents", use_container_width=True, disabled=not uploaded_files):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(uploaded_files)
                if raw_text.strip():
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("Documents processed!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("No text found in uploaded files.")

        st.markdown("---")
        st.subheader("2. Actions")
        
        # Summarize Button
        if st.button("📝 Summarize PDF", use_container_width=True, disabled=not st.session_state.faiss_ready):
            with st.spinner("Summarizing..."):
                try:
                    embeddings = get_embedding_model()
                    # Dangerous deserialization enabled because we created the index ourselves locally
                    vector_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
                    
                    # Retrieve generic context for summary
                    docs = vector_store.similarity_search("Main topics and summary", k=10)
                    
                    template = """
                    You are a helpful assistant. Provide a concise bullet-point summary of the following content.
                    
                    Content:
                    {context}
                    
                    Summary:
                    """
                    chain = get_qa_chain(template)
                    if chain:
                        response = chain.invoke({"context": docs})
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response, 
                            "sources": [d.page_content[:200] for d in docs]
                        })
                        save_chat_history()
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("---")
        st.subheader("3. Session")
        
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.source_toggle = {}
            save_chat_history()
            st.rerun()

        if st.session_state.messages:
            data = format_chat_history(st.session_state.messages)
            st.download_button("Download Chat", data, file_name="chat_history.txt", mime="text/plain", use_container_width=True)

        if st.session_state.faiss_ready and st.button("Delete Index", use_container_width=True):
            shutil.rmtree(FAISS_DIR, ignore_errors=True)
            st.session_state.faiss_ready = False
            st.rerun()

    # --- Main Chat Interface ---
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="font-size: 3em; font-weight: 700; color: #FFFFFF;">
                <span style="margin-right: 15px;">📄</span>Chat With Your Documents
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render Chat
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        
        # Source Toggle Logic
        if msg["role"] == "assistant" and "sources" in msg:
            if st.button("📄 View Context", key=f"src_{idx}"):
                st.session_state.source_toggle[idx] = not st.session_state.source_toggle.get(idx, False)
            
            if st.session_state.source_toggle.get(idx, False):
                with st.expander("Reference Context", expanded=True):
                    for i, source in enumerate(msg["sources"]):
                        st.markdown(f"**Source {i+1}:**\n{source}...")

    # Chat Input
    prompt_placeholder = "Ask a question..." if st.session_state.faiss_ready else "Please upload and process documents first."
    if prompt := st.chat_input(prompt_placeholder, disabled=not st.session_state.faiss_ready):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    embeddings = get_embedding_model()
                    vector_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
                    
                    # Semantic Search
                    docs = vector_store.similarity_search(prompt, k=5)

                    # Modern Chain Execution
                    qa_template = """
                    Answer the question strictly based on the provided context below.
                    If the answer is not in the context, say "The answer is not available in the provided documents."
                    
                    Context:
                    {context}
                    
                    Question: {input}
                    """
                    
                    chain = get_qa_chain(qa_template)
                    if chain:
                        # invoke() is the new standard call for chains
                        response = chain.invoke({"context": docs, "input": prompt})
                        
                        st.markdown(response)
                        
                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response, 
                            "sources": [d.page_content[:200] for d in docs]
                        })
                        save_chat_history()
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
