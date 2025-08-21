#!/usr/bin/env python3
"""
Quick test for the Chat API
Simple script to test if /chat endpoint is working
"""

import requests
import json

# Configuration
BASE_URL = "https://madentk-agents-api-653276357733.me-central1.run.app"

def quick_test():
    print("🚀 Quick Chat API Test")
    print(f"🎯 Testing: {BASE_URL}")
    
    # Test 1: Health check
    try:
        print("\n1️⃣ Testing /health...")
        health = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"   Status: {health.status_code}")
        if health.status_code == 200:
            print(f"   Response: {health.json()}")
        else:
            print(f"   Error: {health.text}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 2: Debug endpoint
    try:
        print("\n2️⃣ Testing /debug...")
        debug = requests.get(f"{BASE_URL}/debug", timeout=10)
        print(f"   Status: {debug.status_code}")
        if debug.status_code == 200:
            debug_info = debug.json()
            print(f"   Chat Router: {debug_info.get('chat_router_status')}")
            print(f"   Total Routes: {debug_info.get('total_routes')}")
            print("   Available Routes:")
            for route in debug_info.get('routes', []):
                print(f"     {route['methods']} {route['path']}")
            print("   Import Status:")
            for module, status in debug_info.get('import_status', {}).items():
                print(f"     {module}: {status}")
        else:
            print(f"   Error: {debug.text}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Test 3: Chat endpoint
    try:
        print("\n3️⃣ Testing /chat...")
        payload = {
            "user_query": "مرحبا",
            "history": [],
            "user_id": None,
            "where": "quweisna"
        }
        
        print(f"   Sending: {json.dumps(payload, ensure_ascii=False)}")
        chat = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)
        print(f"   Status: {chat.status_code}")
        
        if chat.status_code == 200:
            response = chat.json()
            print(f"   ✅ Success!")
            print(f"   Response: {json.dumps(response, ensure_ascii=False, indent=2)}")
        else:
            print(f"   ❌ Failed!")
            print(f"   Error: {chat.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")

if __name__ == "__main__":
    quick_test()
