#!/usr/bin/env python3
"""
Test OpenAI API integration using the key from aci.env
"""

import os
from pathlib import Path

import requests


# Function to load environment variables from aci.env
def load_env(env_path="aci.env"):
    if not Path(env_path).exists():
        print(f"❌ {env_path} not found!")
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()


# Load env vars
load_env()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key or api_key.startswith("your_"):
    print("❌ OPENAI_API_KEY not set in aci.env!")
    exit(1)

print("🔑 Using OpenAI API key:", api_key[:8] + "..." + api_key[-4:])

url = "https://api.openai.com/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
data = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Say hello from my MLB betting system!"}],
}

try:
    print("🧪 Sending test request to OpenAI API...")
    response = requests.post(url, headers=headers, json=data, timeout=20)
    if response.status_code == 200:
        result = response.json()
        print("✅ OpenAI API is working!")
        print("AI says:", result["choices"][0]["message"]["content"].strip())
    else:
        print(f"❌ OpenAI API error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Error: {e}")
