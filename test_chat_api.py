#!/usr/bin/env python3
"""
Test script for the Chat API on Cloud Run
Tests the /chat endpoint with various scenarios
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "https://madentk-agents-api-653276357733.me-central1.run.app"
TIMEOUT = 30

def test_endpoint(method: str, endpoint: str, data: Dict[Any, Any] = None) -> Dict[str, Any]:
    """Test a specific endpoint and return results"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        print(f"\n🔍 Testing {method} {url}")
        
        if method.upper() == "GET":
            response = requests.get(url, timeout=TIMEOUT)
        elif method.upper() == "POST":
            print(f"   Payload: {json.dumps(data, indent=2)}")
            response = requests.post(url, json=data, timeout=TIMEOUT)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        # Response details
        result = {
            "status_code": response.status_code,
            "success": response.status_code < 400,
            "headers": dict(response.headers),
            "url": url,
            "method": method
        }
        
        # Try to parse JSON response
        try:
            result["json"] = response.json()
        except:
            result["text"] = response.text[:500]  # Limit text length
        
        # Log results
        if result["success"]:
            print(f"   ✅ Status: {response.status_code}")
            if "json" in result:
                print(f"   📄 Response: {json.dumps(result['json'], indent=2, ensure_ascii=False)}")
        else:
            print(f"   ❌ Status: {response.status_code}")
            print(f"   📄 Error: {result.get('text', result.get('json', 'No response body'))}")
        
        return result
        
    except requests.exceptions.Timeout:
        print(f"   ⏰ Timeout after {TIMEOUT}s")
        return {"error": "timeout", "url": url}
    except requests.exceptions.ConnectionError as e:
        print(f"   🔌 Connection Error: {e}")
        return {"error": "connection_error", "url": url, "details": str(e)}
    except Exception as e:
        print(f"   💥 Unexpected Error: {e}")
        return {"error": "unexpected", "url": url, "details": str(e)}

def main():
    """Run all tests"""
    print("🚀 Starting Chat API Tests")
    print(f"🎯 Target: {BASE_URL}")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n📋 Test 1: Health Check")
    health_result = test_endpoint("GET", "/health")
    
    # Test 2: Root Endpoint
    print("\n📋 Test 2: Root Endpoint")
    root_result = test_endpoint("GET", "/")
    
    # Test 3: Simple Chat Request
    print("\n📋 Test 3: Simple Chat Request")
    simple_chat = {
        "user_query": "مرحبا",
        "history": [],
        "user_id": None,
        "where": "quweisna"
    }
    chat_result_1 = test_endpoint("POST", "/chat", simple_chat)
    
    # Test 4: Chat with History
    print("\n📋 Test 4: Chat with History")
    chat_with_history = {
        "user_query": "ما هي المطاعم المتاحة؟",
        "history": [
            ["مرحبا", "أهلا وسهلا! كيف يمكنني مساعدتك؟"]
        ],
        "user_id": "test_user_123",
        "where": "quweisna"
    }
    chat_result_2 = test_endpoint("POST", "/chat", chat_with_history)
    
    # Test 5: Empty Query (Error Test)
    print("\n📋 Test 5: Empty Query (Error Test)")
    empty_query = {
        "user_query": "",
        "history": [],
        "user_id": None,
        "where": "quweisna"
    }
    chat_result_3 = test_endpoint("POST", "/chat", empty_query)
    
    # Test 6: Food Order Request
    print("\n📋 Test 6: Food Order Request")
    food_order = {
        "user_query": "أريد طلب بيتزا من مطعم جيد",
        "history": [],
        "user_id": "user_456",
        "where": "quweisna"
    }
    chat_result_4 = test_endpoint("POST", "/chat", food_order)
    
    # Test 7: Invalid Endpoint (404 Test)
    print("\n📋 Test 7: Invalid Endpoint (404 Test)")
    invalid_result = test_endpoint("GET", "/invalid-endpoint")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print("=" * 60)
    
    tests = [
        ("Health Check", health_result),
        ("Root Endpoint", root_result),
        ("Simple Chat", chat_result_1),
        ("Chat with History", chat_result_2),
        ("Empty Query", chat_result_3),
        ("Food Order", chat_result_4),
        ("Invalid Endpoint", invalid_result)
    ]
    
    for test_name, result in tests:
        status = "✅ PASS" if result.get("success") else "❌ FAIL"
        error = f" ({result.get('error', result.get('status_code', 'unknown'))})" if not result.get("success") else ""
        print(f"{status} {test_name}{error}")
    
    # Detailed analysis
    successful_tests = sum(1 for _, result in tests if result.get("success"))
    total_tests = len(tests)
    
    print(f"\n📈 Results: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed! The API is working correctly.")
    elif successful_tests > 0:
        print("⚠️  Some tests failed. Check the details above.")
    else:
        print("🚨 All tests failed. The API may not be working.")
    
    # Specific chat endpoint analysis
    chat_tests = [chat_result_1, chat_result_2, chat_result_3, chat_result_4]
    chat_successes = sum(1 for result in chat_tests if result.get("success"))
    
    if chat_successes == 0:
        print("\n🔥 CRITICAL: /chat endpoint is not working at all!")
        print("   Possible issues:")
        print("   - Route not registered")
        print("   - Import failures")
        print("   - Server not running")
        print("   - Wrong URL")
    elif chat_successes < len(chat_tests):
        print(f"\n⚠️  /chat endpoint partially working ({chat_successes}/{len(chat_tests)} tests passed)")
    else:
        print("\n✅ /chat endpoint is working correctly!")

if __name__ == "__main__":
    main()
