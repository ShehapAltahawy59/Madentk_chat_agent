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
    
    
   
    user_id = st.text_input("User ID (optional)")
    where_value = st.text_input("Where (optional)")
    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("chat_history", None)
        st.rerun()

# Initialize chat history: List[List[Optional[str]]] of [user, assistant]
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[List[Optional[str]]] = []

st.title("SmartFoodAgent Chat")

# Render existing conversation
for user_msg, assistant_msg in st.session_state.chat_history:
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
        "history": st.session_state.chat_history,
    }
    if user_id:
        payload["user_id"] = user_id
    if where_value:
        payload["where"] = where_value
    else:
        payload["where"] = "quweisna"

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
    st.session_state.chat_history.append([prompt, assistant_text]) 
