#!/usr/bin/env python3
"""
Zephyr NFL AI Integration
=========================

Optional Zephyr model interface tailored for the NFL unified system.
This module provides a lightweight async client that can fetch
game-level insight from a hosted Zephyr endpoint while gracefully
handling outages or missing credentials.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ZephyrConfig:
    """Runtime configuration for the Zephyr client."""

    api_key: Optional[str]
    base_url: Optional[str]
    model: str = "zephyr-7b-beta"
    request_timeout: int = 20
    temperature: float = 0.25
    max_tokens: int = 640


class ZephyrNFLAdvisor:
    """
    Async client for Zephyr-powered NFL insights.

    The advisor accepts raw game metadata and returns structured signals
    that can be merged into the unified NFL workflow. When credentials
    are missing or the Zephyr API fails, the advisor returns an empty
    payload so the rest of the pipeline can proceed unaffected.
    """

    def __init__(self, *, config: Optional[ZephyrConfig] = None) -> None:
        env_config = ZephyrConfig(
            api_key=os.getenv("ZEPHYR_API_KEY"),
            base_url=os.getenv("ZEPHYR_BASE_URL"),
            model=os.getenv("ZEPHYR_MODEL", "zephyr-7b-beta"),
            request_timeout=int(os.getenv("ZEPHYR_TIMEOUT", "20")),
            temperature=float(os.getenv("ZEPHYR_TEMPERATURE", "0.25")),
            max_tokens=int(os.getenv("ZEPHYR_MAX_TOKENS", "640")),
        )
        if config:
            env_config = config

        self.config = env_config
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        if not self.is_enabled:
            logger.info("Zephyr NFL Advisor disabled (missing API key or base URL)")

    @property
    def is_enabled(self) -> bool:
        """Return True when the client has enough credentials to run."""
        return bool(self.config.api_key and self.config.base_url)

    async def close(self) -> None:
        """Close the internal HTTP session when it exists."""
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None

    async def analyze_game(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve Zephyr-driven insight for the supplied game data.

        Returns a dictionary with keys:
            - summary: natural language synopsis
            - recommended_action: STRONG BET / MONITOR / PASS etc.
            - confidence: float 0-1
            - risk_flags: list of textual caveats
        """
        if not self.is_enabled:
            return {}

        try:
            session = await self._get_session()
            payload = self._build_payload(game_data)
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }

            async with session.post(
                self.config.base_url,
                headers=headers,
                json=payload,
                timeout=self.config.request_timeout,
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    logger.warning("Zephyr API returned %s: %s", response.status, body[:256])
                    return {}

                data = await response.json()
                return self._parse_response(data)
        except asyncio.TimeoutError:
            logger.warning("Zephyr API request timed out")
        except aiohttp.ClientError as exc:
            logger.warning("Zephyr API request failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected Zephyr integration failure: %s", exc)

        return {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a shared aiohttp session, lazily created."""
        async with self._session_lock:
            if self._session and not self._session.closed:
                return self._session

            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout + 5)
            self._session = aiohttp.ClientSession(timeout=timeout)
            return self._session

    def _build_payload(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Construct the request payload for the Zephyr chat API."""
        system_prompt = (
            "You are an elite NFL betting strategist. Provide concise, actionable insight "
            "based strictly on the supplied game metadata. Include confidence (0-1) and note "
            "risk factors such as injuries, weather, or market volatility when relevant."
        )

        user_prompt = json.dumps(
            {
                "game_id": game_data.get("game_id"),
                "matchup": f"{game_data.get('away_team')} @ {game_data.get('home_team')}",
                "spread": game_data.get("spread"),
                "total": game_data.get("total"),
                "public_percentage": game_data.get("public_percentage"),
                "sharp_percentage": game_data.get("sharp_percentage"),
                "line_movement": game_data.get("line_movement"),
                "edge_potential": game_data.get("edge_potential"),
                "conference": game_data.get("conference"),
                "sport_type": game_data.get("sport_type", "nfl"),
            },
            indent=2,
            sort_keys=True,
        )

        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Analyze the following NFL matchup and recommend a betting posture:\n{user_prompt}",
                },
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

    def _parse_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a Zephyr chat completion payload into structured data.

        Supports the OpenAI-compatible response format commonly returned
        by Zephyr deployments. Any parsing errors fall back to returning
        the raw text under `summary`.
        """
        try:
            choices = payload.get("choices") or []
            if not choices:
                logger.debug("Zephyr payload missing choices: %s", payload)
                return {}

            message = choices[0].get("message", {})
            text = message.get("content", "").strip()
            if not text:
                return {}

            parsed: Dict[str, Any] = {
                "summary": text,
                "recommended_action": "MONITOR",
                "confidence": 0.55,
                "risk_flags": [],
            }

            lowered = text.lower()
            if "strong" in lowered and "bet" in lowered:
                parsed["recommended_action"] = "STRONG BET"
            elif "pass" in lowered:
                parsed["recommended_action"] = "PASS"
            elif "lean" in lowered or "small" in lowered:
                parsed["recommended_action"] = "SMALL BET"

            # Extract primitive confidence cues
            if "confidence" in lowered:
                try:
                    # naive search for floating point confidence values
                    for token in lowered.replace("%", "").split():
                        if token.replace(".", "", 1).isdigit():
                            value = float(token)
                            parsed["confidence"] = value / 100 if value > 1 else value
                            break
                except ValueError:
                    pass

            if "injury" in lowered:
                parsed["risk_flags"].append("INJURY_CONCERN")
            if "weather" in lowered:
                parsed["risk_flags"].append("WEATHER_RISK")
            if "market" in lowered or "line movement" in lowered:
                parsed["risk_flags"].append("MARKET_VOLATILITY")

            return parsed
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to parse Zephyr payload: %s", exc, exc_info=True)
            return {"summary": payload}


__all__ = ["ZephyrNFLAdvisor", "ZephyrConfig"]
