#!/usr/bin/env python3
"""
Quick test of the chat endpoint
"""
import requests
import json

def test_chat():
    url = "http://localhost:8000/api/v1/chat/"
    
    payload = {
        "message": "Hello, how are you?"
    }
    
    try:
        print("Testing chat endpoint...")
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_chat()