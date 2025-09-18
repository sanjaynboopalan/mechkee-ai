import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('OPENAI_API_KEY')
print(f"API Key loaded: {api_key[:10]}... (length: {len(api_key)})")

# Test OpenAI client
try:
    client = openai.OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Say 'Hello, this is a test!'"}
        ],
        max_tokens=50
    )
    
    print("OpenAI Response:", response.choices[0].message.content)
    print("SUCCESS: OpenAI API is working!")
    
except Exception as e:
    print(f"ERROR: {e}")