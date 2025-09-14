#!/usr/bin/env python3
"""
Debug n8n Webhook
================
Test your n8n MLB-Odds-Robot webhook to see what it returns
"""

import asyncio
import json

import aiohttp


async def test_webhook():
    webhook_url = "https://himselfjesus710.app.n8n.cloud/webhook/7716b9d3-1a3b-48f4-a1a1-144d26a94e82"

    print("🤖 Testing n8n webhook...")
    print(f"URL: {webhook_url}")

    async with aiohttp.ClientSession() as session:
        try:
            payload = {"date": "2025-06-22"}
            print(f"Sending payload: {json.dumps(payload, indent=2)}")

            async with session.post(webhook_url, json=payload, timeout=30) as response:
                print(f"\n📊 Response Status: {response.status}")
                print(f"Response Headers: {dict(response.headers)}")

                # Get response text
                text = await response.text()
                print(f"\n📝 Raw Response Text:")
                print("-" * 50)
                print(text)
                print("-" * 50)

                if response.status == 200:
                    try:
                        # Try to parse as JSON
                        data = json.loads(text)
                        print(f"\n✅ Valid JSON Data:")
                        print(json.dumps(data, indent=2))

                        # Check if it has the expected structure
                        if "games" in data:
                            print(
                                f"\n🎯 Found 'games' field with {len(data['games'])} games"
                            )
                        else:
                            print("\n⚠️ No 'games' field found in response")

                        if "h2h_bets" in data:
                            print(
                                f"🎯 Found 'h2h_bets' field with {len(data['h2h_bets'])} bets"
                            )
                        else:
                            print("⚠️ No 'h2h_bets' field found in response")

                    except json.JSONDecodeError as e:
                        print(f"\n❌ Response is not valid JSON: {e}")
                        print("This might be HTML or plain text instead of JSON")

                        # Check if it's HTML (error page)
                        if text.strip().startswith("<"):
                            print(
                                "🌐 Response appears to be HTML (probably an error page)"
                            )

                elif response.status == 422:
                    print(
                        "❌ 422 Unprocessable Entity - JSON parsing error on n8n side"
                    )
                    print(
                        "Your n8n workflow webhook is receiving the request but can't parse it"
                    )

                elif response.status == 404:
                    print(
                        "❌ 404 Not Found - Webhook URL might be incorrect or workflow inactive"
                    )

                else:
                    print(f"❌ Unexpected status code: {response.status}")

        except TimeoutError:
            print("❌ Request timed out after 30 seconds")
            print("Your n8n workflow might be taking too long to process")

        except Exception as e:
            print(f"❌ Request failed: {e}")
            print("Check your internet connection and n8n workflow status")


async def test_simple_get():
    """Test if the webhook endpoint exists with a simple GET request"""
    webhook_url = "https://himselfjesus710.app.n8n.cloud/webhook/7716b9d3-1a3b-48f4-a1a1-144d26a94e82"

    print("\n🔍 Testing webhook with GET request (to check if endpoint exists)...")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(webhook_url, timeout=10) as response:
                print(f"GET Response Status: {response.status}")
                text = await response.text()
                print(f"GET Response: {text[:200]}...")

        except Exception as e:
            print(f"GET Request failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting n8n webhook debug test...")
    asyncio.run(test_webhook())
    asyncio.run(test_simple_get())
    print("\n✅ Debug test completed!")
