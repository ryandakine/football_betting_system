#!/usr/bin/env python3
"""
n8n Data Fetcher v2.2 (Gold Standard - Final)
A high-performance client to fetch structured market data directly
from your n8n automation workflows via webhooks.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta  # <<< --- THIS IS THE FIX ---
from typing import Any, Dict, List

import aiohttp
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# --- Data Structures to Match Your System ---
@dataclass
class GameInfo:
    id: str
    sport_key: str
    commence_time: str
    home_team: str
    away_team: str


@dataclass
class H2HBet:
    game_id: str
    bookmaker: str
    team: str
    odds: float


@dataclass
class StructuredOdds:
    games: list[GameInfo] = field(default_factory=list)
    h2h_bets: list[H2HBet] = field(default_factory=list)
    prop_bets: list[Any] = field(default_factory=list)


# --- The n8n Data Fetcher ---
class N8NDataFetcher:
    """
    Fetches comprehensive market data by triggering a remote n8n workflow.
    """

    def __init__(self, n8n_webhook_url: str):
        if not n8n_webhook_url or not n8n_webhook_url.startswith("http"):
            raise ValueError("A valid n8n webhook URL is required.")
        self.webhook_url = n8n_webhook_url
        self.session = None
        logger.info(f"🤖 n8n Data Fetcher initialized for webhook.")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(2))
    async def fetch_all_markets(self, target_date: date) -> StructuredOdds:
        """
        Triggers the n8n workflow and waits for it to return the final JSON data.
        """
        logger.info(f"▶️ Triggering n8n workflow for date: {target_date.isoformat()}...")
        payload = {"date": target_date.isoformat()}

        try:
            async with self.session.post(self.webhook_url, json=payload) as response:
                response.raise_for_status()
                n8n_response_list = await response.json()
                if not n8n_response_list or "json" not in n8n_response_list[0]:
                    raise ValueError("n8n workflow returned empty or malformed data.")

                structured_data = n8n_response_list[0]["json"]
                logger.info(f"✅ Successfully received data packet from n8n workflow.")
                return self._parse_n8n_response(structured_data)

        except Exception as e:
            logger.error(f"❌ Failed to get data from n8n workflow: {e}", exc_info=True)
            return StructuredOdds()

    def _parse_n8n_response(self, response_data: dict) -> StructuredOdds:
        """
        Parses the structured JSON from n8n into our system's dataclasses.
        """
        try:
            games = [GameInfo(**g) for g in response_data.get("games", [])]
            h2h_bets = [H2HBet(**b) for b in response_data.get("h2h_bets", [])]
            return StructuredOdds(games=games, h2h_bets=h2h_bets)
        except TypeError as e:
            logger.error(f"Mismatch between n8n output and dataclass structure: {e}")
            return StructuredOdds()


async def main():
    """Demonstrates how to use the N8NDataFetcher in your main system."""
    logging.basicConfig(level=logging.INFO)

    load_dotenv()

    n8n_webhook_url = os.getenv("N8N_ODDS_WORKFLOW_URL")

    if not n8n_webhook_url:
        logger.error("N8N_ODDS_WORKFLOW_URL not set in .env file. Exiting.")
        return

    async with N8NDataFetcher(n8n_webhook_url=n8n_webhook_url) as fetcher:
        target_date = datetime.now().date() - timedelta(days=1)
        market_data = await fetcher.fetch_all_markets(target_date=target_date)

    if market_data.games:
        print("\n--- ✅ n8n Integration Test Successful! ---")
        print(f"Fetched data for {len(market_data.games)} games.")
        print(f"Found {len(market_data.h2h_bets)} H2H betting opportunities.")
    else:
        print("\n--- ❌ n8n Integration Test Failed ---")
        print("Could not fetch data. Check your n8n workflow and webhook URL.")


if __name__ == "__main__":
    asyncio.run(main())
