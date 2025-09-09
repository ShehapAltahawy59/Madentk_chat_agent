#!/bin/bash

# Chat Agent Startup Script
echo "Starting Chat Agent..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set environment variables
export CHROMA_DB_DIR=/data/chroma_db

# Start the FastAPI application
echo "Starting FastAPI application on port 8001..."
uvicorn app:app --host 0.0.0.0 --port 8001 --timeout-keep-alive 300
