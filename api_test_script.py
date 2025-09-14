#!/usr/bin/env python3
"""
Simple API Test Script for Fixed MLB System
"""

import asyncio
import os
import ssl
from pathlib import Path

import certifi

import aiohttp


def load_env():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


async def test_claude_api():
    """Test Claude API"""
    print("🔍 Testing Claude API...")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        return False

    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say 'API test successful'"}],
    }

    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["content"][0]["text"]
                    print(f"✅ Claude API: SUCCESS - {content}")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ Claude API: Failed with status {response.status}")
                    print(f"   Response: {text[:200]}...")
                    return False
    except Exception as e:
        print(f"❌ Claude API: Exception - {e}")
        return False


async def test_openai_api():
    """Test OpenAI API"""
    print("🔍 Testing OpenAI API...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Say 'API test successful'"}],
        "max_tokens": 50,
    }

    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    print(f"✅ OpenAI API: SUCCESS - {content}")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ OpenAI API: Failed with status {response.status}")
                    print(f"   Response: {text[:200]}...")
                    return False
    except Exception as e:
        print(f"❌ OpenAI API: Exception - {e}")
        return False


async def test_grok_api():
    """Test Grok API"""
    print("🔍 Testing Grok API...")

    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        print("❌ GROK_API_KEY not found")
        return False

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "grok-beta",
        "messages": [{"role": "user", "content": "Say 'API test successful'"}],
        "max_tokens": 50,
    }

    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            async with session.post(
                "https://api.x.ai/v1/chat/completions", headers=headers, json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    print(f"✅ Grok API: SUCCESS - {content}")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ Grok API: Failed with status {response.status}")
                    print(f"   Response: {text[:200]}...")
                    return False
    except Exception as e:
        print(f"❌ Grok API: Exception - {e}")
        return False


async def test_odds_api():
    """Test Odds API"""
    print("🔍 Testing Odds API...")

    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        print("❌ THE_ODDS_API_KEY not found")
        return False

    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            url = "https://api.the-odds-api.com/v4/sports"
            params = {"apiKey": api_key}

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    sport_count = len(data)
                    print(f"✅ Odds API: SUCCESS - Found {sport_count} sports")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ Odds API: Failed with status {response.status}")
                    print(f"   Response: {text[:200]}...")
                    return False
    except Exception as e:
        print(f"❌ Odds API: Exception - {e}")
        return False


async def main():
    """Run all API tests"""
    print("🚀 API Connection Test Suite")
    print("=" * 50)

    # Load environment variables
    load_env()

    # Test all APIs
    tests = [
        ("Claude", test_claude_api()),
        ("OpenAI", test_openai_api()),
        ("Grok", test_grok_api()),
        ("Odds API", test_odds_api()),
    ]

    results = {}
    for name, test_coro in tests:
        try:
            results[name] = await test_coro
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results[name] = False
        print()  # Add spacing

    # Summary
    print("📊 TEST SUMMARY")
    print("=" * 50)
    successful_tests = 0
    total_tests = len(results)

    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:12} : {status}")
        if success:
            successful_tests += 1

    print(f"\nOverall: {successful_tests}/{total_tests} APIs working")

    if successful_tests >= 3:
        print("🎉 Sufficient APIs working - system should run successfully!")
        print("💡 You can now run: python fixed_launcher.py")
    elif successful_tests >= 1:
        print("⚠️  Some APIs working - system will run with fallbacks")
        print("💡 You can still run: python fixed_launcher.py")
    else:
        print("🚨 No APIs working - check your API keys and internet connection")

    return successful_tests


if __name__ == "__main__":
    asyncio.run(main())
