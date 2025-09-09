#!/bin/bash

# Chat Agent Streamlit UI Startup Script
echo "Starting Chat Agent Streamlit UI..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set environment variables
export CHAT_API_BASE_URL=${CHAT_API_BASE_URL:-http://173.212.251.191:8001}

# Start the Streamlit application
echo "Starting Streamlit application on port 8502..."
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8502
