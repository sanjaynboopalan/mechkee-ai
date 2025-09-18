#!/usr/bin/env python3
"""
Test the API endpoints to verify they're working
"""

import requests
import json
import time

def test_api_endpoints():
    """Test all API endpoints"""
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 Testing Perplexity AI Clone API Endpoints...")
    print("=" * 50)
    
    try:
        # Test 1: Root endpoint
        print("1. Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        print(f"   ✅ Root endpoint: {response.status_code}")
        print(f"   📋 Response: {response.json()}")
        
        # Test 2: Health check
        print("\n2. Testing health endpoint...")
        response = requests.get(f"{base_url}/api/v1/health/")
        print(f"   ✅ Health endpoint: {response.status_code}")
        print(f"   📋 Response: {response.json()}")
        
        # Test 3: Search endpoint
        print("\n3. Testing search endpoint...")
        search_data = {
            "query": "What is artificial intelligence?",
            "max_results": 5,
            "include_citations": True,
            "search_type": "hybrid"
        }
        response = requests.post(f"{base_url}/api/v1/search/", json=search_data)
        print(f"   ✅ Search endpoint: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   📋 Answer length: {len(result.get('answer', ''))}")
            print(f"   📋 Sources found: {len(result.get('sources', []))}")
            print(f"   📋 Citations: {len(result.get('citations', []))}")
            print(f"   📋 Search time: {result.get('search_time', 0):.3f}s")
        else:
            print(f"   ❌ Error: {response.text}")
        
        # Test 4: Chat endpoint
        print("\n4. Testing chat endpoint...")
        chat_data = {
            "message": "Hello, can you help me understand machine learning?",
            "session_id": None,
            "context_limit": 5
        }
        response = requests.post(f"{base_url}/api/v1/chat/", json=chat_data)
        print(f"   ✅ Chat endpoint: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   📋 Session ID: {result.get('session_id', 'N/A')}")
            print(f"   📋 Response length: {len(result.get('message', {}).get('content', ''))}")
        else:
            print(f"   ❌ Error: {response.text}")
        
        # Test 5: Documents list
        print("\n5. Testing documents endpoint...")
        response = requests.get(f"{base_url}/api/v1/documents/")
        print(f"   ✅ Documents endpoint: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   📋 Documents found: {len(result)}")
        else:
            print(f"   ❌ Error: {response.text}")
        
        print("\n🎉 All API tests completed!")
        print("\n📍 API is accessible at:")
        print(f"   • Main API: {base_url}")
        print(f"   • Documentation: {base_url}/docs")
        print(f"   • Health Check: {base_url}/api/v1/health/")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the server is running on http://127.0.0.1:8000")
        return False
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Wait a moment for server to be ready
    time.sleep(2)
    test_api_endpoints()