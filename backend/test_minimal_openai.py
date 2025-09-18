import openai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

print(f"Testing API Key: {api_key[:15]}...")

client = openai.OpenAI(api_key=api_key)

try:
    # Try the cheapest model first
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print("✅ SUCCESS! API key works!")
    print("Response:", response.choices[0].message.content)
    
except Exception as e:
    print("❌ ERROR:", e)
    if "429" in str(e):
        print("\n💡 SOLUTION: Add credits to your OpenAI billing account")
        print("   Go to: https://platform.openai.com/account/billing")
    elif "401" in str(e):
        print("\n💡 SOLUTION: Check your API key")
        print("   Go to: https://platform.openai.com/api-keys")