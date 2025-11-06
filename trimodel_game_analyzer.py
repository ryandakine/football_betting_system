# trimodel_game_analyzer.py

# ✅ CRITICAL FIX: Load environment variables FIRST before any imports
from dotenv import load_dotenv

load_dotenv()  # This ensures .env file is loaded before API calls

import asyncio
import json
import logging
import re
import statistics
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Your central configuration is now the single source of truth
from tri_model_api_config import get_ensemble_config, get_trimodel_api_keys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a quantitative sports analyst. Your task is to analyze an MLB game. Your response MUST be a single, valid JSON object with no other text, containing these exact keys: "confidence_score" (a float from 0.50 to 0.95), "favored_team" (a string matching one of the team names), and "reasoning" (a brief 1-2 sentence explanation)."""


class AIClient(ABC):
    """An abstract interface defining what every AI client must be able to do."""

    @abstractmethod
    async def analyze(self, session: aiohttp.ClientSession, game_info: dict) -> dict:
        pass


class ClaudeClient(AIClient):
    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
        self.endpoint = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    @retry(
        wait=wait_exponential(min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def analyze(self, session: aiohttp.ClientSession, game_info: dict) -> dict:
        payload = {
            "model": self.model_id,
            "max_tokens": 300,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": json.dumps(game_info)}],
        }
        async with session.post(
            self.endpoint, headers=self.headers, json=payload, timeout=30
        ) as r:
            r.raise_for_status()
            content = (await r.json())["content"][0]["text"]
            return {**json.loads(content), "source": "Claude"}


class OpenAIClient(AIClient):
    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    @retry(
        wait=wait_exponential(min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def analyze(self, session: aiohttp.ClientSession, game_info: dict) -> dict:
        payload = {
            "model": self.model_id,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(game_info)},
            ],
        }
        async with session.post(
            self.endpoint, headers=self.headers, json=payload, timeout=25
        ) as r:
            r.raise_for_status()
            content = (await r.json())["choices"][0]["message"]["content"]
            return {**json.loads(content), "source": "OpenAI"}


class GrokClient(AIClient):
    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
        self.endpoint = "https://api.x.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    @retry(
        wait=wait_exponential(min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def analyze(self, session: aiohttp.ClientSession, game_info: dict) -> dict:
        prompt = f"{SYSTEM_PROMPT}\n\nGame Data: {json.dumps(game_info)}"
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        async with session.post(
            self.endpoint, headers=self.headers, json=payload, timeout=35
        ) as r:
            r.raise_for_status()
            content = (await r.json())["choices"][0]["message"]["content"]
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                raise ValueError("Grok response did not contain valid JSON.")
            return {**json.loads(json_match.group()), "source": "Grok"}


class TriModelGameAnalyzer:
    """The main analyzer class that uses the corrected clients."""

    def __init__(self):
        logger.info("🤖 Initializing TriModelGameAnalyzer...")
        self.config = get_ensemble_config()
        self.clients = self._initialize_clients()

    def _initialize_clients(self) -> dict[str, AIClient]:
        clients = {}
        api_keys = get_trimodel_api_keys()
        model_ids = self.config["models"]

        if key := api_keys.get("claude_api"):
            clients["claude"] = ClaudeClient(api_key=key, model_id=model_ids["claude"])
        if key := api_keys.get("openai_api"):
            clients["openai"] = OpenAIClient(api_key=key, model_id=model_ids["openai"])
        if key := api_keys.get("grok_api"):
            clients["grok"] = GrokClient(
                api_key=key, model_id=model_ids.get("grok", "grok-2")
            )

        logger.info(
            f"✅ Initialized {len(clients)} real AI clients: {list(clients.keys())}"
        )
        return clients

    async def _analyze_single_game(
        self, session: aiohttp.ClientSession, game: dict
    ) -> dict:
        """Analyzes one game with all models and returns a single ensemble result."""
        tasks = [client.analyze(session, game) for client in self.clients.values()]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = [res for res in responses if not isinstance(res, Exception)]
        if not valid_responses:
            logger.warning(
                f"All AI analyses failed for game {game.get('id', 'unknown')}"
            )
            return {"game_id": game.get("id"), "error": "All AI analyses failed."}

        return self._ensemble_responses(valid_responses, game.get("id"))

    def _ensemble_responses(self, responses: list[dict], game_id: str) -> dict:
        """Combines multiple AI responses into a single, weighted prediction with analytics."""
        weights = self.config["weights"]

        # Weighted average confidence
        weighted_sum = sum(
            res["confidence_score"] * weights[res["source"].lower()]
            for res in responses
        )
        total_weight = sum(weights[res["source"].lower()] for res in responses)
        consensus_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Agreement level (lower is better)
        conf_scores = [r["confidence_score"] for r in responses]
        agreement_std_dev = (
            statistics.stdev(conf_scores) if len(conf_scores) > 1 else 0.0
        )

        # Majority vote on favored team
        teams = [r.get("favored_team") for r in responses if r.get("favored_team")]
        favored_team = max(set(teams), key=teams.count) if teams else "Unknown"

        return {
            "game_id": game_id,
            "consensus_confidence": round(consensus_confidence, 4),
            "favored_team": favored_team,
            "agreement_score": round(1 - agreement_std_dev, 4),  # Higher is better
            "model_count": len(responses),
            "individual_analyses": responses,
        }

    async def analyze_games(self, games_list: list[dict]) -> dict[str, dict]:
        """Analyzes a list of games in parallel and returns a dictionary of results."""
        if not games_list:
            return {}
        logger.info(f"🧠 Starting parallel analysis for {len(games_list)} games...")

        async with aiohttp.ClientSession() as session:
            tasks = [self._analyze_single_game(session, game) for game in games_list]
            results = await asyncio.gather(*tasks)

        results_dict = {
            res.get("game_id"): res for res in results if res and "error" not in res
        }
        logger.info(f"✅ AI analysis complete for {len(results_dict)} games.")
        return results_dict
