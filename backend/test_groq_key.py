#!/usr/bin/env python3
"""
Test the Groq API key directly
"""
import groq
import os
from dotenv import load_dotenv

load_dotenv()

def test_groq_key():
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("❌ GROQ_API_KEY not found in environment")
        return
    
    print(f"✅ Found GROQ_API_KEY: {api_key[:10]}...")
    
    try:
        client = groq.Groq(api_key=api_key)
        
        print("🔄 Testing Groq API...")
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Say hello"}
            ],
            model="llama-3.1-8b-instant",
            max_tokens=50,
            timeout=10
        )
        
        print("✅ Groq API working!")
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Groq API error: {e}")

if __name__ == "__main__":
    test_groq_key()