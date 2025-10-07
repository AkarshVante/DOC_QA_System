import streamlit as st
from transformers import pipeline

# --- APPLICATION CONFIGURATION AND STYLING ---

# Set the page configuration, including the title and icon.
st.set_page_config(
    page_title="Solar AI Assistant",
    page_icon="☀️"
)

# Inject custom CSS for UI enhancements using st.markdown.
# This includes styling for sidebar buttons and hiding the main menu and sidebar collapse control.
st.markdown("""
<style>
    /* Add a light blue border to all buttons within the sidebar */
    button {
        border: 2px solid #ADD8E6;
    }

    /* Hide the sidebar collapse control button to make the sidebar permanent */
    div[data-testid="collapsedControl"] {
        visibility: hidden;
    }

    /* Hide the main Streamlit toolbar menu to remove the theme toggle */
    button {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE INITIALIZATION AND HELPER FUNCTIONS ---

@st.cache_resource
def load_model():
    """
    Load the Hugging Face text2text-generation pipeline.
    Uses @st.cache_resource to load the model only once per session.
    The API token is fetched securely from st.secrets.
    """
    try:
        token = st.secrets
        text2text_generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            use_auth_token=token
        )
        return text2text_generator
    except (KeyError, FileNotFoundError):
        st.error("Hugging Face API token not found. Please add it to your secrets.toml file.")
        return None
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return None

def handle_ask_ai():
    """
    Callback function for the 'Ask AI' button.
    This function is executed before the script reruns, ensuring immediate UI updates.
    """
    user_prompt = st.session_state.get("user_prompt", "").strip()
    if user_prompt and st.session_state.text2text_generator:
        # Append user's message to the chat history.
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Generate a response from the AI model.
        with st.spinner("AI is thinking..."):
            try:
                response = st.session_state.text2text_generator(user_prompt)
                # Correctly parse the response from the Hugging Face pipeline structure.
                bot_reply = response['generated_text']
                # Append AI's response to the chat history.
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error."})
        
        # Clear the input box after submission.
        st.session_state.user_prompt = ""

# Initialize session state variables if they don't exist.
if "text2text_generator" not in st.session_state:
    st.session_state.text2text_generator = load_model()

if "messages" not in st.session_state:
    st.session_state.messages =

# --- SIDEBAR UI ---

with st.sidebar:
    st.title("☀️ Solar AI Assistant")
    st.info("This is a chatbot powered by Google's Flan-T5 model. Ask any question and get a response from the AI.")
    
    # The 'New Chat' button clears the message history.
    if st.button("New Chat"):
        st.session_state.messages =
        st.rerun() # Use st.rerun to immediately reflect the cleared chat.

# --- MAIN CHAT INTERFACE ---

# Display the chat history by iterating through messages in session state.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display the chat input widget at the bottom of the page.
# The 'on_click' callback is used for robust state handling.
st.chat_input(
    "Ask a question...",
    key="user_prompt",
    on_submit=handle_ask_ai,
    disabled=not st.session_state.text2text_generator # Disable input if model failed to load
)

# Add a placeholder to handle the initial empty state.
if not st.session_state.messages:
    st.info("Start the conversation by typing a message below.")
