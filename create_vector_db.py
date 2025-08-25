#!/usr/bin/env python3
"""
Script to create and populate the vector database for semantic search.
This script fetches data from Firebase and creates embeddings for restaurants and items.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🚀 Vector Database Creation Script")
    print("=" * 50)
    
    try:
        # Import the tools module
        from agent.tools import recreate_vector_database
        
        print("🔄 Starting vector database creation...")
        result = recreate_vector_database()
        
        if result["status"] == "success":
            print("✅ " + result["message"])
            print("\n🎉 Vector database created successfully!")
            print("You can now use semantic search in your application.")
        else:
            print("❌ " + result["message"])
            print("\n💡 Troubleshooting tips:")
            print("1. Check your Firebase credentials")
            print("2. Ensure you have data in your Firestore collections")
            print("3. Check your internet connection for model downloads")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.backend.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
