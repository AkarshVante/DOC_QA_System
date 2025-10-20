# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import shutil
# import time
# import json

# # --- Configuration ---
# # Load environment variables from .env file for local development
# load_dotenv()

# # Set Streamlit page configuration
# st.set_page_config(page_title="ChatPDF", layout="wide", initial_sidebar_state="expanded")

# # --- UI Styling (from app1.py) ---
# # Custom CSS for the dark, modern UI with WhatsApp-like chat messages
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

#     /* Define animation */
#     @keyframes fadeIn {
#         from { opacity: 0; transform: translateY(10px); }
#         to { opacity: 1; transform: translateY(0); }
#     }
    
#     html, body, [class*="st-"] {
#         font-family: 'Inter', sans-serif;
#     }
    
#     /* Hide other default Streamlit elements */
#     #MainMenu, footer, .stDeployButton {
#         visibility: hidden;
#     }
    
#     /* Main app background */
#     .stApp {
#         background-color: #07101a;
#     }
    
#     /* --- Sidebar Styling --- */
#     [data-testid="stSidebar"] {
#         background-color: #07101a;
#         border-right: 1px solid #13303f;
#     }
    
#     /* Sidebar button (st.button & st.download_button) with glowing effect */
#     [data-testid="stSidebar"] .stButton button, 
#     [data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
#         border-radius: 999px;
#         border: 1px solid #2c5970;
#         background-color: transparent;
#         color: #add8e6;
#         transition: all 0.2s ease-in-out;
#         box-shadow: 0 0 5px 0px rgba(0, 150, 255, 0.3);
#     }
#     [data-testid="stSidebar"] .stButton button:hover,
#     [data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
#         background-color: rgba(173, 216, 230, 0.1);
#         color: #fff;
#         border-color: #00aaff;
#         box-shadow: 0 0 10px 2px rgba(0, 150, 255, 0.6);
#     }
    
#     /* Status badge styling */
#     .status-badge {
#         display: block; padding: 8px; border-radius: 20px;
#         font-weight: 600; margin: 12px auto; text-align: center;
#     }
#     .status-ready { background-color: rgba(25, 195, 125, 0.1); color: #19c37d; }
#     .status-not-ready { background-color: rgba(255, 102, 51, 0.1); color: #ff6633; }
    
#     /* --- Main Chat Interface Styling --- */
#     .stChatMessage {
#         animation: fadeIn 0.5s ease-out;
#         transition: all 0.2s ease-in-out;
#     }
    
#     /* The actual bubble inside the container */
#     div[data-testid="stChatMessage"] div[data-testid^="stMarkdownContainer"] {
#         border-radius: 12px;
#         padding: 14px 18px;
#         margin: 4px;
#         color: white;
#         border: 1px solid #2c5970;
#         box-shadow: 0 0 8px 1px rgba(0, 150, 255, 0.15);
#     }

#     /* Assistant message bubble styling */
#     div[data-testid="stChatMessage-assistant"] div[data-testid^="stMarkdownContainer"] {
#         background-color: #262D31; /* Dark grey, like WhatsApp dark mode */
#         border-bottom-left-radius: 4px; /* WhatsApp tail effect */
#     }

#     /* User message bubble styling - REMOVED background color */
#     div[data-testid="stChatMessage-user"] div[data-testid^="stMarkdownContainer"] {
#         background-color: transparent; 
#         border-bottom-right-radius: 4px; /* WhatsApp tail effect */
#     }
    
#     /* Chat input box styling */
#     [data-testid="stChatInput"] textarea {
#         color: #FFFFFF; /* Makes the input text white and visible */
#         max-height: 150px; /* Make chat input box smaller */
#     }

#     /* Style for the "View Source" button to be small and outside the chat box */
#     .stButton>button {
#         background-color: transparent !important;
#         color: #add8e6 !important;
#         border: 1px solid #2c5970 !important;
#         padding: 2px 10px !important; /* Smaller padding */
#         font-size: 0.8rem !important; /* Smaller font */
#         border-radius: 999px !important;
#         margin: -8px 0 10px 10px; /* Positioning below the chat bubble */
#     }
#     .stButton>button:hover {
#         border-color: #00aaff !important;
#         color: #fff !important;
#     }

# </style>
# """, unsafe_allow_html=True)


# # --- Google API Configuration ---
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     st.error("Google API key not found. Please set it in Streamlit secrets or as an environment variable.")
#     st.stop()

# try:
#     genai.configure(api_key=GOOGLE_API_KEY)
# except Exception as e:
#     st.error(f"Failed to configure Google Generative AI: {e}")
#     st.stop()


# # --- Constants ---
# # Define the preferred model order, with fallbacks for the generative part
# MODEL_ORDER = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# FAISS_DIR = "faiss_index"
# CHAT_HISTORY_FILE = "chat_history.json"


# # --- Helper Functions ---

# def get_pdf_text(pdf_docs):
#     """Extract text from a list of uploaded PDF files."""
#     text = ""
#     for pdf in pdf_docs:
#         try:
#             pdf_reader = PdfReader(pdf)
#             # Add metadata for source highlighting
#             for i, page in enumerate(pdf_reader.pages):
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += f"\n--- Source: {pdf.name}, Page: {i+1} ---\n{page_text}"
#         except Exception as e:
#             st.warning(f"Could not read file: {pdf.name}. Error: {e}")
#     return text

# def get_text_chunks(text):
#     """Split text into manageable chunks for processing."""
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks

# @st.cache_resource
# def get_embedding_model():
#     """Load the sentence transformer model, cached for performance."""
#     return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# def get_vector_store(text_chunks):
#     """Create and save a FAISS vector store from text chunks using a local model."""
#     try:
#         embeddings = get_embedding_model()
#         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#         vector_store.save_local(FAISS_DIR)
#         st.session_state.faiss_ready = True
#     except Exception as e:
#         st.error(f"Error creating vector store: {e}")
#         st.session_state.faiss_ready = False

# def get_conversational_chain(prompt_template):
#     """Create a conversational QA chain with a custom prompt and model fallback."""
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
#     for model_name in MODEL_ORDER:
#         try:
#             model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
#             chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#             st.session_state.last_model_used = model_name
#             return chain
#         except Exception:
#             st.warning(f"Model {model_name} not available. Trying next model.")
    
#     st.error("All specified Gemini models are unavailable. Please check your API key and model access.")
#     return None

# def format_chat_history(messages):
#     """Formats the chat history for downloading."""
#     chat_str = "Chat History\n"
#     chat_str += "="*20 + "\n\n"
#     for msg in messages:
#         role = "User" if msg["role"] == "user" else "Assistant"
#         content = msg['content']
#         chat_str += f"[{role}]:\n{content}\n\n"
#         chat_str += "-"*20 + "\n\n"
#     return chat_str

# def save_chat_history():
#     """Save chat history to a JSON file."""
#     with open(CHAT_HISTORY_FILE, "w") as f:
#         json.dump(st.session_state.messages, f)

# def load_chat_history():
#     """Load chat history from a JSON file if it exists."""
#     if os.path.exists(CHAT_HISTORY_FILE):
#         with open(CHAT_HISTORY_FILE, "r") as f:
#             return json.load(f)
#     return []

# # --- Main Application ---
# def main():
#     """Main function to run the Streamlit application."""

#     # Initialize session state variables
#     if "messages" not in st.session_state:
#         st.session_state.messages = load_chat_history()
#     if "faiss_ready" not in st.session_state:
#         st.session_state.faiss_ready = os.path.isdir(FAISS_DIR)
#     if "source_toggle" not in st.session_state:
#         st.session_state.source_toggle = {}


#     # --- Sidebar for Document and Session Management ---
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
#             type=['pdf'],
#             label_visibility="collapsed"
#         )
        
#         if st.button("2. Process Documents", use_container_width=True, disabled=not uploaded_files):
#             with st.spinner("Processing documents... This may take a moment."):
#                 raw_text = get_pdf_text(uploaded_files)
#                 if raw_text.strip():
#                     chunks = get_text_chunks(raw_text)
#                     get_vector_store(chunks)
#                     st.success("✅ Documents processed!")
#                     time.sleep(1)
#                     st.rerun()
#                 else:
#                     st.error("Processing failed. No readable text found in PDFs.")

#         st.markdown("---")
#         st.subheader("2. Advanced Options")
#         if st.button("📝 Summarize PDF", use_container_width=True, disabled=not st.session_state.faiss_ready):
#             with st.spinner("Summarizing document..."):
#                 try:
#                     embeddings = get_embedding_model()
#                     vector_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
#                     docs = vector_store.similarity_search("Summarize the entire document", k=len(vector_store.index_to_docstore_id))
                    
#                     # A specific prompt for summarization
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
#                 use_container_width=True
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
            
#     # --- Main Chat Interface ---
#     st.markdown("""
#     <div style="text-align: center; margin-bottom: 20px;">
#         <h1 style="font-size: 3em; font-weight: 700; color: #FFFFFF;">
#             <span style="margin-right: 15px;">📄</span>Chat With Your Documents
#         </h1>
#     </div>
#     """, unsafe_allow_html=True)

#     # Display chat messages
#     for idx, msg in enumerate(st.session_state.messages):
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])
        
#         # Place button and source info OUTSIDE the chat bubble for assistants
#         if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
#             if st.button("📄 View Source", key=f"src_{idx}"):
#                 st.session_state.source_toggle[idx] = not st.session_state.source_toggle.get(idx, False)
            
#             if st.session_state.source_toggle.get(idx, False):
#                 st.info("".join(msg["sources"]))


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
                    
#                     # Standard QA prompt
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
#                         answer = "The conversation chain could not be initialized. Please check the logs."
                        
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




import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import shutil
import time
import json

# --- Configuration ---
# Load environment variables from .env file for local development
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="ChatPDF", layout="wide", initial_sidebar_state="expanded")

# --- UI Styling (from app1.py) ---
# Custom CSS for the dark, modern UI with WhatsApp-like chat messages
st.markdown("""
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
    
    /* Sidebar button (st.button & st.download_button) with glowing effect */
    [data-testid="stSidebar"] .stButton button, 
    [data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
        border-radius: 999px;
        border: 1px solid #2c5970;
        background-color: transparent;
        color: #add8e6;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 0 5px 0px rgba(0, 150, 255, 0.3);
    }
    [data-testid="stSidebar"] .stButton button:hover,
    [data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
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
    .stChatMessage {
        animation: fadeIn 0.5s ease-out;
        transition: all 0.2s ease-in-out;
    }
    
    /* The actual bubble inside the container */
    div[data-testid="stChatMessage"] div[data-testid^="stMarkdownContainer"] {
        border-radius: 12px;
        padding: 14px 18px;
        margin: 4px;
        color: white;
        border: 1px solid #2c5970;
        box-shadow: 0 0 8px 1px rgba(0, 150, 255, 0.15);
    }

    /* Assistant message bubble styling */
    div[data-testid="stChatMessage-assistant"] div[data-testid^="stMarkdownContainer"] {
        background-color: #262D31; /* Dark grey, like WhatsApp dark mode */
        border-bottom-left-radius: 4px; /* WhatsApp tail effect */
    }

    /* User message bubble styling - REMOVED background color */
    div[data-testid="stChatMessage-user"] div[data-testid^="stMarkdownContainer"] {
        background-color: transparent; 
        border-bottom-right-radius: 4px; /* WhatsApp tail effect */
    }
    
    /* Chat input box styling */
    [data-testid="stChatInput"] textarea {
        color: #FFFFFF; /* Makes the input text white and visible */
        max-height: 150px; /* Make chat input box smaller */
    }

    /* Style for the "View Source" button specifically within the main chat area */
    .main-chat-area .stButton>button {
        background-color: transparent !important;
        color: #add8e6 !important;
        border: 1px solid #2c5970 !important;
        padding: 2px 10px !important; /* Smaller padding */
        font-size: 0.8rem !important; /* Smaller font */
        border-radius: 999px !important;
        margin: -8px 0 10px 10px; /* Positioning below the chat bubble */
    }
    .main-chat-area .stButton>button:hover {
        border-color: #00aaff !important;
        color: #fff !important;
    }

</style>
""", unsafe_allow_html=True)


# --- Google API Configuration ---
# Use Streamlit secrets for deployed apps, fallback to .env for local development
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set it in Streamlit secrets or as a local .env file.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {e}")
    st.stop()


# --- Constants ---
# Define the preferred model order, with fallbacks for the generative part
MODEL_ORDER = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_DIR = "faiss_index"
CHAT_HISTORY_FILE = "chat_history.json"


# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    """Extract text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            # Add metadata for source highlighting
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Source: {pdf.name}, Page: {i+1} ---\n{page_text}"
        except Exception as e:
            st.warning(f"Could not read file: {pdf.name}. Error: {e}")
    return text

def get_text_chunks(text):
    """Split text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_embedding_model():
    """Load the sentence transformer model, cached for performance."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_vector_store(text_chunks):
    """Create and save a FAISS vector store from text chunks using a local model."""
    try:
        embeddings = get_embedding_model()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_DIR)
        st.session_state.faiss_ready = True
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.session_state.faiss_ready = False

def get_conversational_chain(prompt_template):
    """Create a conversational QA chain with a custom prompt and model fallback."""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    for model_name in MODEL_ORDER:
        try:
            model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            st.session_state.last_model_used = model_name
            return chain
        except Exception:
            st.warning(f"Model {model_name} not available. Trying next model.")
    
    st.error("All specified Gemini models are unavailable. Please check your API key and model access.")
    return None

def format_chat_history(messages):
    """Formats the chat history for downloading."""
    chat_str = "Chat History\n"
    chat_str += "="*20 + "\n\n"
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg['content']
        chat_str += f"[{role}]:\n{content}\n\n"
        chat_str += "-"*20 + "\n\n"
    return chat_str

def save_chat_history():
    """Save chat history to a JSON file."""
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(st.session_state.messages, f)

def load_chat_history():
    """Load chat history from a JSON file if it exists."""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()
    if "faiss_ready" not in st.session_state:
        st.session_state.faiss_ready = os.path.isdir(FAISS_DIR)
    # This dictionary will hold the visibility state for each source button
    if "source_toggle" not in st.session_state:
        st.session_state.source_toggle = {}


    # --- Sidebar for Document and Session Management ---
    with st.sidebar:
        st.header("📄 ChatPDF")
        st.markdown("Your personal document assistant.")
        
        status_text = "Ready" if st.session_state.faiss_ready else "No Documents"
        status_class = "status-ready" if st.session_state.faiss_ready else "status-not-ready"
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
            with st.spinner("Processing documents... This may take a moment."):
                raw_text = get_pdf_text(uploaded_files)
                if raw_text.strip():
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("✅ Documents processed!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Processing failed. No readable text found in PDFs.")

        st.markdown("---")
        st.subheader("2. Advanced Options")
        if st.button("📝 Summarize PDF", use_container_width=True, disabled=not st.session_state.faiss_ready):
            with st.spinner("Summarizing document..."):
                try:
                    embeddings = get_embedding_model()
                    vector_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
                    docs = vector_store.similarity_search("Summarize the entire document", k=len(vector_store.index_to_docstore_id))
                    
                    summary_prompt_template = """
                    Based on the following context, provide a concise summary in bullet points.
                    Focus on the key topics, findings, and conclusions.\n\n
                    Context:\n {context}\n
                    Question: \n{question}\n

                    Summary:
                    """
                    chain = get_conversational_chain(summary_prompt_template)
                    if chain:
                        response = chain({"input_documents": docs, "question": "Summarize the entire document."}, return_only_outputs=True)
                        summary = response.get("output_text", "Could not generate a summary.")
                        st.session_state.messages.append({"role": "assistant", "content": summary, "sources": [doc.page_content[:150] + "..." for doc in docs]})
                        save_chat_history()
                        st.rerun()

                except Exception as e:
                    st.error(f"An error occurred during summarization: {e}")

        st.markdown("---")
        st.subheader("3. Manage Session")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.source_toggle = {}
            save_chat_history()
            st.rerun()

        if st.session_state.messages:
            chat_history_str = format_chat_history(st.session_state.messages)
            st.download_button(
                label="Download Chat",
                data=chat_history_str,
                file_name=f"chat_history_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        if st.session_state.faiss_ready and st.button("Delete Documents", use_container_width=True):
            shutil.rmtree(FAISS_DIR, ignore_errors=True)
            st.session_state.faiss_ready = False
            st.session_state.messages = []
            st.session_state.source_toggle = {}
            save_chat_history()
            st.success("Documents and index deleted.")
            time.sleep(1)
            st.rerun()
            
    # --- Main Chat Interface ---
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="font-size: 3em; font-weight: 700; color: #FFFFFF;">
            <span style="margin-right: 15px;">📄</span>Chat With Your Documents
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Wrap chat area in a div to scope the button style
    st.markdown('<div class="main-chat-area">', unsafe_allow_html=True)

    # Display chat messages
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        
        # Render the button and source info OUTSIDE the chat bubble for assistants
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            if st.button("📄 View Source", key=f"src_{idx}"):
                # Toggle the visibility state for this specific message index
                st.session_state.source_toggle[idx] = not st.session_state.source_toggle.get(idx, False)
            
            # Conditionally display the source info based on the toggle state
            if st.session_state.source_toggle.get(idx, False):
                st.info("".join(msg["sources"]))

    # Chat input
    prompt_placeholder = "Please process documents first..." if not st.session_state.faiss_ready else "Ask a question..."
    if prompt := st.chat_input(prompt_placeholder, disabled=not st.session_state.faiss_ready):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Rerun to show the user's message immediately
        st.rerun()

    # Handle assistant response generation if the last message is from the user
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    last_user_prompt = st.session_state.messages[-1]['content']
                    embeddings = get_embedding_model()
                    vector_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
                    docs = vector_store.similarity_search(last_user_prompt, k=4)
                    
                    qa_prompt_template = """
                    Answer the question as detailed as possible from the provided context. If the answer is not in
                    the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.\n\n
                    Context:\n {context}\n
                    Question: \n{question}\n

                    Answer:
                    """
                    chain = get_conversational_chain(qa_prompt_template)
                    if chain:
                        response = chain({"input_documents": docs, "question": last_user_prompt}, return_only_outputs=True)
                        answer = response.get("output_text", "Sorry, I couldn't generate a response.")
                    else:
                        answer = "The conversation chain could not be initialized. Please check the logs."
                    
                    sources = [doc.page_content[:150] + "..." for doc in docs]
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    save_chat_history()
                    # Rerun to display the new assistant message and its source button
                    st.rerun()

                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    save_chat_history()
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()





