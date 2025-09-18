#!/usr/bin/env python3

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Test Groq API connection
def test_groq():
    try:
        # Load API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        print(f"API Key loaded: {api_key[:20]}..." if api_key else "No API key found")
        
        if not api_key:
            print("❌ GROQ_API_KEY not found in environment variables")
            return False
        
        # Initialize Groq client
        client = Groq(api_key=api_key)
        print("✅ Groq client initialized")
        
        # Test a simple completion with different models
        models_to_try = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant", 
            "llama-3.2-3b-preview",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768"
        ]
        
        for model in models_to_try:
            try:
                print(f"Trying model: {model}")
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": "Say hello in exactly 5 words"}
                    ],
                    max_tokens=50,
                    temperature=0.1
                )
                
                answer = response.choices[0].message.content
                print(f"✅ Success with {model}!")
                print(f"Response: {answer}")
                return model  # Return the working model
                
            except Exception as e:
                print(f"❌ Failed with {model}: {str(e)}")
                continue
        
        return None
        
    except Exception as e:
        print(f"❌ Groq API test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_groq()