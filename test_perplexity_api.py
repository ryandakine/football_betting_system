#!/usr/bin/env python3
"""
Test Perplexity Pro API integration using the key from aci.env
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

api_key = os.getenv("PERPLEXITY_API_KEY")

if not api_key or api_key.startswith("your_"):
    print("❌ PERPLEXITY_API_KEY not set in aci.env!")
    exit(1)

print("🔑 Using Perplexity Pro API key:", api_key[:8] + "..." + api_key[-4:])

# Try different Perplexity API endpoints
endpoints = [
    "https://api.perplexity.ai/chat/completions",
    "https://api.perplexity.ai/v1/chat/completions",
    "https://api.perplexity.ai/completions",
]

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

# Try different model names
models = [
    "llama-3.1-sonar-large-128k-online",
    "llama-3.1-sonar-large-128k",
    "sonar-large-online",
    "gpt-4",
    "claude-3.5-sonar",
]

data = {
    "model": "llama-3.1-sonar-large-128k-online",
    "messages": [{"role": "user", "content": "Say hello from my MLB betting system!"}],
    "max_tokens": 500,
}

print("🧪 Testing Perplexity Pro API...")

for endpoint in endpoints:
    print(f"  Trying endpoint: {endpoint}")
    try:
        response = requests.post(endpoint, headers=headers, json=data, timeout=10)
        print(f"    Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Perplexity Pro API is working!")
            print("AI says:", result["choices"][0]["message"]["content"].strip())
            break
        elif response.status_code == 401:
            print("    ❌ 401 Unauthorized - Invalid API key")
        elif response.status_code == 404:
            print("    ❌ 404 Not Found - Wrong endpoint")
        else:
            print(f"    ❌ Error {response.status_code}")

    except Exception as e:
        print(f"    ❌ Connection error: {e}")
else:
    print("\n❌ Could not connect to Perplexity Pro API")
    print("Possible issues:")
    print("1. API key might be invalid")
    print("2. API endpoint might have changed")
    print("3. Network connectivity issues")
    print("\nPlease check your Perplexity Pro API key and try again.")
