"""
Direct test for OpenAI API key and chat completion
"""
import os
from dotenv import load_dotenv, find_dotenv

# Load .env with override
load_dotenv(find_dotenv(), override=True)

from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
print(f"OPENAI_API_KEY loaded: {api_key[:8]}... (length: {len(api_key) if api_key else 0})")

if not api_key:
    print("❌ No OPENAI_API_KEY found!")
    exit(1)

try:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=10
    )
    print("✅ OpenAI API call successful!")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print(f"❌ OpenAI API call failed: {e}") 