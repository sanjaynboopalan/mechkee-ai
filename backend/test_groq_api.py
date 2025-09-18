#!/usr/bin/env python3
"""
Test script for Groq API connectivity
"""

import os
import sys
from groq import Groq

def test_groq_api():
    """Test Groq API connectivity"""
    try:
        # Load API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            print("❌ GROQ_API_KEY not found in environment")
            return False
        
        print("🔑 Testing Groq API...")
        print(f"API Key: {api_key[:20]}...")
        
        # Initialize client
        client = Groq(api_key=api_key)
        
        # Test with different models
        models_to_try = [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile", 
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
        
        for model in models_to_try:
            try:
                print(f"🔍 Trying model: {model}")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hello, this is a test. Please respond with 'API working correctly'."}],
                    max_tokens=50,
                    temperature=0.1
                )
                
                print(f"✅ {model} working!")
                print(f"Response: {response.choices[0].message.content}")
                return True
                
            except Exception as e:
                print(f"❌ {model} failed: {e}")
                continue
        
    except Exception as e:
        print(f"❌ Groq API failed: {e}")
        return False

if __name__ == "__main__":
    test_groq_api()