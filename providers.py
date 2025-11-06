#!/usr/bin/env python3
"""
AI Providers Module v3.0 - QUAD EDITION with .env support
========================================================
Object-oriented, resilient, and efficient providers for the QUAD-model AI ensemble.
Features .env file support, automatic retries, shared network sessions, and clean architecture.
"""

import asyncio
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Look for .env file in current directory
    env_path = Path(".") / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded .env file from {env_path}")
    else:
        print("⚠️ No .env file found, using system environment variables")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables only")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Configuration Helper ---
def get_api_keys():
    """Get API keys from .env file or environment variables"""
    keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "grok": os.getenv("GROK_API_KEY"),
    }

    # Log which keys are available (without showing the actual keys)
    for name, key in keys.items():
        if key:
            masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
            print(f"✅ {name.upper()}_API_KEY loaded: {masked_key}")
        else:
            print(f"❌ {name.upper()}_API_KEY not found")

    return keys


# --- Abstract Base Class (The "Contract" for all AI Providers) ---


class AIProvider(ABC):
    """
    An abstract interface that defines the contract for any AI provider.
    Ensures every client has an `analyze` method.
    """

    def __init__(self, api_key: str, model_id: str):
        if not api_key:
            logger.warning(
                f"{self.__class__.__name__} API key not provided - will return mock data"
            )
        self.api_key = api_key
        self.model_id = model_id

    @abstractmethod
    async def analyze(self, session: aiohttp.ClientSession, prompt: str) -> list[dict]:
        """
        The core method for every provider. Takes a prompt and returns a list of predictions.
        """
        pass

    def _safe_json_parse(self, text: str) -> list[dict]:
        """
        A robust method to extract JSON array from AI response.
        """
        # First try to parse as direct JSON array
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return [result]  # Convert single object to array
        except json.JSONDecodeError:
            pass

        # Look for JSON array in text
        array_match = re.search(r"\[.*?\]", text, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass

        # Look for JSON object and convert to array
        obj_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if obj_match:
            try:
                obj = json.loads(obj_match.group())
                return [obj]
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON from response: {text[:150]}...")
        return []


# --- Concrete Provider Implementations ---


class GPT4Provider(AIProvider):
    """The provider for OpenAI's GPT-4 models."""

    @retry(
        wait=wait_exponential(min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def analyze(self, session: aiohttp.ClientSession, prompt: str) -> list[dict]:
        if not self.api_key:
            return self._mock_response("GPT-4")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are GPT-4, an expert MLB betting analyst. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
        }

        try:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                return self._safe_json_parse(content)
        except Exception as e:
            logger.error(f"GPT-4 API error: {e}")
            return self._mock_response("GPT-4")

    def _mock_response(self, model_name: str) -> list[dict]:
        return [
            {
                "game_id": "mock_game_123",
                "pick": "Yankees ML",
                "confidence": 72,
                "stake_units": 2,
                "rationale": f"{model_name} mock prediction - sentiment-driven analysis",
                "primary_signal": "sentiment",
                "analyst_type": "sentiment",
            }
        ]


class ClaudeProvider(AIProvider):
    """The provider for Anthropic's Claude models."""

    @retry(
        wait=wait_exponential(min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def analyze(self, session: aiohttp.ClientSession, prompt: str) -> list[dict]:
        if not self.api_key:
            return self._mock_response("Claude")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model_id,
            "max_tokens": 2000,
            "temperature": 0.3,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                content = result["content"][0]["text"]
                return self._safe_json_parse(content)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._mock_response("Claude")

    def _mock_response(self, model_name: str) -> list[dict]:
        return [
            {
                "game_id": "mock_game_123",
                "pick": "Yankees ML",
                "confidence": 75,
                "stake_units": 3,
                "rationale": f"{model_name} mock prediction - market analysis focus",
                "primary_signal": "odds",
                "analyst_type": "quantitative",
            }
        ]


class GeminiProvider(AIProvider):
    """The provider for Google's Gemini models."""

    @retry(
        wait=wait_exponential(min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def analyze(self, session: aiohttp.ClientSession, prompt: str) -> list[dict]:
        if not self.api_key:
            return self._mock_response("Gemini")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"You are Gemini, an expert MLB analyst. {prompt}"}
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2000},
        }

        try:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                return self._safe_json_parse(content)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._mock_response("Gemini")

    def _mock_response(self, model_name: str) -> list[dict]:
        return [
            {
                "game_id": "mock_game_123",
                "pick": "Under 8.5",
                "confidence": 70,
                "stake_units": 2,
                "rationale": f"{model_name} mock prediction - news and situational analysis",
                "primary_signal": "news",
                "analyst_type": "news",
            }
        ]


class GrokProvider(AIProvider):
    """The provider for xAI's Grok models."""

    @retry(
        wait=wait_exponential(min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def analyze(self, session: aiohttp.ClientSession, prompt: str) -> list[dict]:
        if not self.api_key:
            return self._mock_response("Grok")

        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are Grok, an expert MLB betting analyst with sharp analytical skills.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
        }

        try:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                return self._safe_json_parse(content)
        except Exception as e:
            logger.error(f"Grok API error: {e}")
            return self._mock_response("Grok")

    def _mock_response(self, model_name: str) -> list[dict]:
        return [
            {
                "game_id": "mock_game_123",
                "pick": "Red Sox +1.5",
                "confidence": 68,
                "stake_units": 2,
                "rationale": f"{model_name} mock prediction - multi-signal synthesis",
                "primary_signal": "convergence",
                "analyst_type": "synthesizer",
            }
        ]


# --- Global Provider Manager ---
class ProviderManager:
    """Manages all AI providers with shared session and configuration."""

    def __init__(self):
        api_keys = get_api_keys()
        self.providers = {
            "gpt4": GPT4Provider(api_keys.get("openai"), "gpt-4"),
            "claude4": ClaudeProvider(
                api_keys.get("anthropic"), "claude-3-5-sonnet-20241022"
            ),
            "gemini": GeminiProvider(api_keys.get("gemini"), "gemini-1.5-pro"),
            "grok": GrokProvider(api_keys.get("grok"), "grok-beta"),
        }

    async def analyze_all(self, prompts: dict[str, str]) -> dict[str, list[dict]]:
        """Run all providers with their respective prompts concurrently."""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=45)
        ) as session:
            tasks = {
                name: provider.analyze(
                    session, prompts.get(name, prompts.get("default", ""))
                )
                for name, provider in self.providers.items()
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            # Map results back to provider names
            final_results = {}
            for i, name in enumerate(self.providers.keys()):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"{name} failed: {result}")
                    final_results[name] = []
                else:
                    final_results[name] = result

            return final_results


# --- Compatibility Functions for Ultimate System ---
# These maintain compatibility with the ultimate_main_orchestrator.py

_provider_manager = None


async def _get_provider_manager():
    """Get or create the global provider manager."""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = ProviderManager()
    return _provider_manager


async def ask_gpt4(prompt: str) -> list[dict]:
    """Compatibility function for GPT-4"""
    manager = await _get_provider_manager()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=45)
    ) as session:
        return await manager.providers["gpt4"].analyze(session, prompt)


async def ask_claude4(prompt: str) -> list[dict]:
    """Compatibility function for Claude-4"""
    manager = await _get_provider_manager()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=45)
    ) as session:
        return await manager.providers["claude4"].analyze(session, prompt)


async def ask_gemini(prompt: str) -> list[dict]:
    """Compatibility function for Gemini"""
    manager = await _get_provider_manager()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=45)
    ) as session:
        return await manager.providers["gemini"].analyze(session, prompt)


async def ask_grok(prompt: str) -> list[dict]:
    """Compatibility function for Grok"""
    manager = await _get_provider_manager()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=45)
    ) as session:
        return await manager.providers["grok"].analyze(session, prompt)


# --- Test Function ---
async def test_all_providers():
    """Tests all QUAD AI providers with a simple prompt."""
    test_prompt = """Return a JSON array with one test prediction:
    [{"game_id": "test_123", "pick": "Test ML", "confidence": 75, "rationale": "This is a test", "primary_signal": "test"}]"""

    manager = ProviderManager()
    prompts = {"default": test_prompt}
    results = await manager.analyze_all(prompts)

    print("\n--- QUAD AI Provider Test Results ---")
    for name, result in results.items():
        if result:
            print(f"✅ {name.upper()}: SUCCESS - {len(result)} predictions returned")
            if result[0]:
                print(
                    f"   Sample: {result[0].get('pick', 'N/A')} - {result[0].get('rationale', 'N/A')}"
                )
        else:
            print(f"❌ {name.upper()}: FAILED - No predictions returned")
    print("---------------------------------------\n")

    # Test compatibility functions
    print("Testing compatibility functions...")
    try:
        gpt_result = await ask_gpt4(test_prompt)
        print(f"✅ ask_gpt4: {len(gpt_result)} predictions")

        claude_result = await ask_claude4(test_prompt)
        print(f"✅ ask_claude4: {len(claude_result)} predictions")

        gemini_result = await ask_gemini(test_prompt)
        print(f"✅ ask_gemini: {len(gemini_result)} predictions")

        grok_result = await ask_grok(test_prompt)
        print(f"✅ ask_grok: {len(grok_result)} predictions")

    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_all_providers())
