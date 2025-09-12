import os
import requests
import streamlit as st
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables for local dev
load_dotenv()

st.set_page_config(page_title="SmartFoodAgent Chat", page_icon="🍽️", layout="centered")

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
    
    
   
    user_id = st.text_input("User ID (optional)", value="dIbhODNvagNACfhKWZy3")
    
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
    if st.button("Clear chat", use_container_width=True):
        # Clear history for current session only
        st.session_state.chat_histories[current_session] = []
        st.rerun()

# Initialize session-based chat history
if "session_id" not in st.session_state:
    st.session_state.session_id = st.session_state.get("session_id", f"session_{hash(str(st.session_state))}")
    
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

# Get or create chat history for current session
current_session = st.session_state.session_id
if current_session not in st.session_state.chat_histories:
    st.session_state.chat_histories[current_session] = []

chat_history = st.session_state.chat_histories[current_session]

# Track location changes
if "current_location" not in st.session_state:
    st.session_state.current_location = where_value
elif st.session_state.current_location != where_value:
    st.session_state.current_location = where_value
    # Clear history for current session when location changes
    st.session_state.chat_histories[current_session] = []
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
    st.sidebar.text(f"Session: {current_session[:8]}...")
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

    # Append the full pair to history after receiving the response
    chat_history.append([prompt, assistant_text]) 
