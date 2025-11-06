#!/usr/bin/env python3
"""
Fix API Authentication Issues for Football Betting System
Tests and validates all API connections
"""

import asyncio
import aiohttp
import os
from api_config import get_api_keys

async def test_all_apis():
    """Test all API connections and provide fixes"""
    
    print("🔧 TESTING API CONNECTIONS...\n")
    
    api_keys = get_api_keys()
    results = {}
    
    # Test Claude
    print("Testing Claude API...")
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "x-api-key": api_keys["claude"],
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "claude-3-sonnet-20240229",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                results["Claude"] = f"✅ Working ({resp.status})"
    except Exception as e:
        results["Claude"] = f"❌ Error: {str(e)[:50]}"
    
    # Test OpenAI
    print("Testing OpenAI API...")
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {api_keys['openai']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                results["OpenAI"] = f"✅ Working ({resp.status})"
    except Exception as e:
        results["OpenAI"] = f"❌ Error: {str(e)[:50]}"
    
    # Test Perplexity
    print("Testing Perplexity API...")
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {api_keys['perplexity']}",
                "Content-Type": "application/json"
            }
            # Updated model name for Perplexity
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            async with session.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 401:
                    results["Perplexity"] = "❌ Invalid API key - Check your PERPLEXITY_API_KEY"
                else:
                    results["Perplexity"] = f"✅ Working ({resp.status})"
    except Exception as e:
        results["Perplexity"] = f"❌ Error: {str(e)[:50]}"
    
    # Test Grok
    print("Testing Grok API...")
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {api_keys['grok']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "grok-beta",  # Try different model names
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            # Try different endpoints
            endpoints = [
                "https://api.x.ai/v1/chat/completions",
                "https://api.grok.x.ai/v1/chat/completions",
                "https://api.x.com/v1/chat/completions"
            ]
            
            for endpoint in endpoints:
                try:
                    async with session.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status < 500:
                            results["Grok"] = f"✅ Working at {endpoint} ({resp.status})"
                            break
                except:
                    continue
            else:
                results["Grok"] = "❌ All Grok endpoints failed - API may be down or key invalid"
    except Exception as e:
        results["Grok"] = f"❌ Error: {str(e)[:50]}"
    
    # Test Gemini
    print("Testing Gemini API...")
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_keys['gemini']}"
            payload = {
                "contents": [{"parts": [{"text": "test"}]}]
            }
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                results["Gemini"] = f"✅ Working ({resp.status})"
    except Exception as e:
        results["Gemini"] = f"❌ Error: {str(e)[:50]}"
    
    # Test Odds API
    print("Testing Odds API...")
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.the-odds-api.com/v4/sports?apiKey={api_keys['odds_api']}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results["Odds API"] = f"✅ Working (Quota: {resp.headers.get('x-requests-remaining', 'Unknown')})"
                else:
                    results["Odds API"] = f"❌ Status: {resp.status}"
    except Exception as e:
        results["Odds API"] = f"❌ Error: {str(e)[:50]}"
    
    return results

def print_results(results):
    """Print test results and recommendations"""
    print("\n" + "="*60)
    print("📊 API STATUS REPORT")
    print("="*60)
    
    working = []
    broken = []
    
    for api, status in results.items():
        print(f"{api:15} {status}")
        if "✅" in status:
            working.append(api)
        else:
            broken.append(api)
    
    print("\n" + "="*60)
    print("📈 SUMMARY")
    print("="*60)
    print(f"✅ Working: {len(working)}/{len(results)} APIs")
    print(f"   {', '.join(working)}")
    
    if broken:
        print(f"\n❌ Need Fixing: {len(broken)} APIs")
        print(f"   {', '.join(broken)}")
        
        print("\n🔧 RECOMMENDATIONS:")
        
        if "Perplexity" in broken:
            print("\n📌 Perplexity Fix:")
            print("   1. Get a new API key from https://www.perplexity.ai/settings/api")
            print("   2. Update your .env file: PERPLEXITY_API_KEY=pplx-xxxxx")
            
        if "Grok" in broken:
            print("\n📌 Grok Fix:")
            print("   1. Grok API may be in beta or changed endpoints")
            print("   2. Check https://x.ai/api for latest documentation")
            print("   3. Consider using fallback to free providers")
    
    print("\n💡 TIP: The system will use free AI providers (HuggingFace, Local) as fallback")
    print("        when premium APIs fail, so you can still run predictions!")

async def main():
    results = await test_all_apis()
    print_results(results)

if __name__ == "__main__":
    asyncio.run(main())
