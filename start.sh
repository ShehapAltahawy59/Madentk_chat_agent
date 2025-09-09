#!/bin/bash

# Chat Agent Startup Script
echo "Starting Chat Agent..."

# Activate virtual environment if it exists
if [ -d "madentk-chat-agent-venv" ]; then
    echo "Activating virtual environment..."
    source madentk-chat-agent-venv/bin/activate
fi

# Set environment variables
export CHROMA_DB_DIR=/data/chroma_db

# Start the FastAPI application
echo "Starting FastAPI application on port 8001..."
uvicorn app:app --host 0.0.0.0 --port 8001 --timeout-keep-alive 300
