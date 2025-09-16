import os
import requests
import streamlit as st
from typing import List, Optional
from dotenv import load_dotenv
from uuid import uuid4

# Load environment variables for local dev
load_dotenv()

st.set_page_config(page_title="SmartFoodAgent Chat", page_icon="🍽️", layout="centered")

# Global in-process chat store to allow persistence across Streamlit sessions
# keyed by a stable identifier (user_id if provided, else a URL sid).
CHAT_STORE = globals().setdefault("CHAT_STORE", {})

# Sidebar configuration
with st.sidebar:
    st.markdown("**Server**")
    # Get default URL from environment or use correct Cloud Run URL
    env_url = os.environ.get("CHAT_API_BASE_URL", "http://localhost:8080")
    
    # If it's an old/incorrect Cloud Run URL, use the correct one
    if "madentk-agents-api-roishx3apa-ww.a.run.app" in env_url:
        default_base_url = "https://madentk-agents-api-653276357733.me-central1.run.app"
        st.info("🔄 Using correct Cloud Run URL")
    else:
        default_base_url = env_url
    
    base_url = default_base_url
    
    # Quick fix button for correct URL
    
    
   
    user_id = st.text_input("User ID (optional)", value="pPpnUSXN9SVDZJaEDYhzzTuFRp92")
    
    # Location selection
    st.markdown("**Location**")
    location_options = ["quweisna", "AboHammad", "KafrShokr"]
    where_value = st.selectbox(
        "Select your location",
        options=location_options,
        index=0,  # Default to quweisna
        help="Choose your delivery location"
    )
    st.markdown("---")
    # Clear button wired later after keys are initialized
    clear_placeholder = st.empty()

# Initialize a stable session id via URL query param `sid` (use new st.query_params API)
sid = st.query_params.get("sid")
if not sid:
    sid = str(uuid4())[:12]
    st.query_params["sid"] = sid

# Initialize containers
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

# Determine the history key: prefer user_id if provided later, else sid
# We'll initialize with sid for now; after user_id is entered we will remap
history_key = sid
if history_key not in st.session_state.chat_histories:
    # Seed from global CHAT_STORE if present
    st.session_state.chat_histories[history_key] = CHAT_STORE.get(history_key, [])

chat_history = st.session_state.chat_histories[history_key]

# Track location changes
if "current_location" not in st.session_state:
    st.session_state.current_location = where_value
elif st.session_state.current_location != where_value:
    st.session_state.current_location = where_value
    # Clear history for current session when location changes
    st.session_state.chat_histories[history_key] = []
    CHAT_STORE[history_key] = []
    st.rerun()

st.title("SmartFoodAgent Chat")

# Show welcome message if no chat history
if not chat_history:
    st.success("🎉 Welcome! You can now order food, search for dishes, and get personalized recommendations. Start by typing your request below!")

st.info(f"📍 Current location: **{where_value}** - Only restaurants and items from this location will be shown")

# Render existing conversation
for user_msg, assistant_msg in chat_history:
    if user_msg:
        with st.chat_message("user"):
            st.markdown(user_msg)
    if assistant_msg:
        with st.chat_message("assistant"):
            st.markdown(assistant_msg)

# Now that we know user_id, if it's filled move the history to be keyed by user_id
if user_id:
    user_key = user_id.strip()
    if user_key and user_key != history_key:
        # Migrate history to new key (user-based)
        CHAT_STORE[user_key] = CHAT_STORE.get(user_key, st.session_state.chat_histories.get(history_key, []))
        st.session_state.chat_histories[user_key] = CHAT_STORE[user_key]
        history_key = user_key
        chat_history = st.session_state.chat_histories[history_key]

# Wire clear button now that history_key is known
with clear_placeholder:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_histories[history_key] = []
        CHAT_STORE[history_key] = []
        st.rerun()

# Chat input
prompt = st.chat_input("اكتب طلبك هنا…")

if prompt:
    # Show the user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare request payload matching FastAPI schema using existing history only
    payload = {
        "user_query": prompt,
        "history": chat_history,
        "where": where_value,  # Always send where value
    }
    if user_id:
        payload["user_id"] = user_id
    
    # Debug: Show what we're sending
    st.sidebar.markdown("**Debug Info**")
    st.sidebar.text(f"Location: {where_value}")
    st.sidebar.text(f"User ID: {user_id or 'None'}")
    st.sidebar.text(f"Session: {history_key[:12]}...")
    st.sidebar.text(f"History length: {len(chat_history)}")

    # Call the /chat endpoint
    try:
        resp = requests.post(f"{base_url.rstrip('/')}/chat", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        assistant_text = data.get("response", "")
    except Exception as e:
        assistant_text = f"حصل خطأ أثناء الاتصال بالسيرفر: {e}"

    # Render assistant response
    with st.chat_message("assistant"):
        st.markdown(assistant_text)

    # Append the full pair to history after receiving the response and persist
    chat_history.append([prompt, assistant_text])
    st.session_state.chat_histories[history_key] = chat_history
    CHAT_STORE[history_key] = chat_history
