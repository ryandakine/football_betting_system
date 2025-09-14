#!/usr/bin/env python3
"""
Enhanced MLB Game Selection with Intelligent Caching
Production-ready AI analysis with cost-efficient caching mechanism
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Required AI libraries
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from api_config import get_api_keys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Complete system prompt with enhanced instructions
SYSTEM_PROMPT = """
You are an expert MLB betting analyst with deep knowledge of baseball statistics, player performance, and game dynamics.

Analyze the provided MLB game and assess the likelihood of the HOME TEAM winning. Consider these factors:
- Starting pitcher matchups (ERA, WHIP, recent performance, opposing team batting averages)
- Team offensive capabilities (runs per game, OPS, recent hot/cold streaks)
- Bullpen strength and usage patterns
- Home field advantage (typically worth ~0.54 win probability)
- Rest days, travel fatigue, and scheduling advantages
- Weather conditions affecting play (wind, temperature, humidity)
- Injuries to key players and lineup changes
- Historical head-to-head performance
- Current team momentum and psychological factors

Provide a confidence score for HOME TEAM victory from 0.50 (no edge, pure coin flip) to 0.95 (maximum confidence).
DO NOT go below 0.50 or above 0.95 under any circumstances.

Return ONLY valid JSON with this exact structure:
{
  "confidence_score": 0.XX,
  "reasoning": "Concise 1-2 sentence explanation for your assessment",
  "home_team_edge": true/false
}

Example Response:
{
  "confidence_score": 0.73,
  "reasoning": "Home team's ace pitcher faces struggling opposing offense while enjoying strong bullpen depth and favorable weather conditions.",
  "home_team_edge": true
}
"""


class GameSelector:
    """
    Enhanced MLB game analysis with intelligent caching and multiple AI providers.
    Features cost-efficient caching, parallel AI analysis, and production-ready error handling.
    """

    def __init__(
        self, alert_manager=None, cache_duration_hours=6, cache_dir="testing_logs"
    ):
        """
        Initialize GameSelector with AI clients and caching capabilities.

        Args:
            alert_manager: Optional alert manager for notifications
            cache_duration_hours: How long to cache analysis results (default: 6 hours)
            cache_dir: Directory to store cache files
        """
        logger.info("🤖 Initializing Enhanced GameSelector with caching...")
        self.alert_manager = alert_manager
        self.clients = {}

        # AI Model Configuration (latest versions)
        self.model_config = {
            "claude": "claude-3-5-sonnet-20241022",  # Latest Claude 3.5 Sonnet
            "openai": "gpt-4o",  # Latest GPT-4 Optimized
        }

        # Caching Configuration
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "ai_analysis_cache.json")
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self._ensure_cache_directory()
        self._load_cache()

        # Initialize AI clients
        self._initialize_ai_clients()

        logger.info(
            f"🤖 GameSelector ready with {len(self.clients)} AI providers and {cache_duration_hours}h caching"
        )

    def _ensure_cache_directory(self):
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _initialize_ai_clients(self):
        """Initialize all available AI clients."""
        try:
            api_keys = get_api_keys()

            # Initialize Claude
            if claude_key := api_keys.get("claude"):
                self.clients["claude"] = AsyncAnthropic(api_key=claude_key)
                logger.info("✅ Claude 3.5 Sonnet client initialized")
            else:
                logger.warning("⚠️ Claude API key not found")

            # Initialize OpenAI
            if openai_key := api_keys.get("openai"):
                self.clients["openai"] = AsyncOpenAI(api_key=openai_key)
                logger.info("✅ GPT-4o client initialized")
            else:
                logger.warning("⚠️ OpenAI API key not found")

            if not self.clients:
                raise ValueError("No valid AI clients could be initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize AI clients: {e}")
            if self.alert_manager:
                asyncio.create_task(
                    self._send_alert(f"AI client initialization failed: {e}")
                )
            self.clients = {}

    # --- Intelligent Caching System ---

    def _load_cache(self):
        """Load the AI analysis cache from persistent storage."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file) as f:
                    self.cache = json.load(f)
                # Clean expired entries
                self._clean_expired_cache()
                logger.info(f"✅ Loaded {len(self.cache)} valid items from AI cache")
            else:
                self.cache = {}
                logger.info("📝 Created new AI analysis cache")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"⚠️ Cache load failed ({e}), starting fresh")
            self.cache = {}

    def _save_cache(self):
        """Save the current cache to persistent storage."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")

    def _clean_expired_cache(self):
        """Remove expired entries from cache."""
        current_time = datetime.now(timezone.utc)
        expired_keys = []

        for game_id, cached_item in self.cache.items():
            try:
                timestamp = datetime.fromisoformat(cached_item["timestamp"])
                if current_time - timestamp > self.cache_duration:
                    expired_keys.append(game_id)
            except (KeyError, ValueError):
                expired_keys.append(game_id)  # Remove malformed entries

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.info(f"🧹 Cleaned {len(expired_keys)} expired cache entries")

    def _get_cache_key(self, game_info: dict) -> str:
        """Generate a unique cache key for a game."""
        game_id = game_info.get("game_id", "unknown")
        # Include commence_time in key to handle postponed games
        commence_time = game_info.get("commence_time", "")
        return f"{game_id}_{commence_time}"

    def _get_from_cache(self, cache_key: str) -> dict | None:
        """Retrieve a valid result from cache if not expired."""
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            try:
                timestamp = datetime.fromisoformat(cached_item["timestamp"])
                if datetime.now(timezone.utc) - timestamp < self.cache_duration:
                    logger.info(f"✅ Using cached analysis for {cache_key}")
                    return cached_item["data"]
            except (KeyError, ValueError):
                # Remove malformed cache entry
                del self.cache[cache_key]

        return None

    def _update_cache(self, cache_key: str, data: dict):
        """Update cache with new analysis result."""
        self.cache[cache_key] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        self._save_cache()

    # --- AI Analysis Engine ---

    def _format_game_data_for_ai(self, game_info: dict) -> dict:
        """Format game data for AI consumption with enhanced context."""
        home_team = game_info.get("home_team", "Unknown")
        away_team = game_info.get("away_team", "Unknown")
        commence_time = game_info.get("commence_time", "")

        # Parse and format game time
        game_time_str = "Unknown"
        day_of_week = "Unknown"
        try:
            if commence_time:
                dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                game_time_str = dt.strftime("%I:%M %p ET")
                day_of_week = dt.strftime("%A")
        except Exception:
            pass

        return {
            "game_id": game_info.get("game_id", "unknown"),
            "matchup": f"{away_team} @ {home_team}",
            "home_team": home_team,
            "away_team": away_team,
            "game_time": game_time_str,
            "day_of_week": day_of_week,
            "commence_time_iso": commence_time,
            "context": f"MLB game analysis: {away_team} visiting {home_team} on {day_of_week} at {game_time_str}",
        }

    async def _execute_ai_analysis(self, provider: str, game_data: dict) -> dict:
        """
        Execute AI analysis for a specific provider with enhanced error handling.
        """
        client = self.clients[provider]
        model = self.model_config[provider]
        prompt_content = f"Analyze this MLB game for HOME TEAM win probability:\n\n{json.dumps(game_data, indent=2)}"

        try:
            if provider == "claude":
                message = await client.messages.create(
                    model=model,
                    max_tokens=250,
                    temperature=0.1,  # Low temperature for consistent analysis
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt_content}],
                )
                response_text = message.content[0].text.strip()

            elif provider == "openai":
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_content},
                    ],
                    max_tokens=250,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                response_text = response.choices[0].message.content.strip()

            else:
                raise ValueError(f"Unknown provider: {provider}")

            # Parse and validate JSON response
            result = json.loads(response_text)

            # Validate and clamp confidence score
            confidence = float(result.get("confidence_score", 0.5))
            confidence = max(0.5, min(0.95, confidence))

            return {
                "provider": f"{provider.capitalize()} ({model})",
                "confidence": confidence,
                "reasoning": result.get("reasoning", "No reasoning provided"),
                "home_team_edge": result.get("home_team_edge", False),
                "raw_response": response_text,
            }

        except json.JSONDecodeError as e:
            logger.error(f"{provider} returned invalid JSON: {e}")
            return {
                "provider": f"{provider.capitalize()} ({model})",
                "confidence": 0.5,
                "reasoning": f"JSON parse error: {str(e)[:100]}",
                "error": "json_parse_error",
            }
        except Exception as e:
            logger.error(f"{provider} analysis failed: {e}")
            return {
                "provider": f"{provider.capitalize()} ({model})",
                "confidence": 0.5,
                "reasoning": f"Analysis failed: {str(e)[:100]}",
                "error": str(e),
            }

    async def _send_alert(self, message: str):
        """Send alert if manager is available."""
        if self.alert_manager:
            try:
                await self.alert_manager.send_alert(
                    alert_type="AI_ANALYSIS_ERROR",
                    severity="HIGH",
                    message=message,
                    details={"component": "GameSelector"},
                )
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

    # --- Main Analysis Methods ---

    async def analyze_game(self, game_info: dict) -> dict:
        """
        Analyze a single game with caching and multiple AI providers.

        Args:
            game_info: Game metadata dictionary

        Returns:
            Analysis result with consensus confidence and individual provider results
        """
        cache_key = self._get_cache_key(game_info)
        game_id = game_info.get("game_id", "unknown")

        # Step 1: Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result

        # Step 2: Handle case with no AI clients
        if not self.clients:
            logger.warning(f"No AI clients available for game {game_id}")
            fallback_result = {
                "game_id": game_id,
                "game_info": game_info,
                "consensus_confidence": 0.51,
                "individual_analyses": [
                    {
                        "provider": "Fallback",
                        "confidence": 0.51,
                        "reasoning": "No AI clients available",
                        "error": "no_clients",
                    }
                ],
                "providers_count": 0,
                "cache_status": "no_cache_fallback",
            }
            return fallback_result

        # Step 3: Format data and run AI analyses
        formatted_data = self._format_game_data_for_ai(game_info)
        logger.info(f"🔍 Analyzing {formatted_data['matchup']} (cache miss)")

        # Run all AI analyses in parallel
        analysis_tasks = [
            self._execute_ai_analysis(provider, formatted_data)
            for provider in self.clients.keys()
        ]

        individual_analyses = await asyncio.gather(*analysis_tasks)

        # Filter out failed analyses
        valid_analyses = [a for a in individual_analyses if "error" not in a]

        if not valid_analyses:
            logger.error(f"All AI analyses failed for game {game_id}")
            await self._send_alert(f"All AI analyses failed for game {game_id}")

            error_result = {
                "game_id": game_id,
                "game_info": game_info,
                "consensus_confidence": 0.5,
                "individual_analyses": individual_analyses,
                "providers_count": 0,
                "cache_status": "analysis_failed",
            }
            return error_result

        # Step 4: Calculate consensus and create result
        confidences = [a["confidence"] for a in valid_analyses]
        consensus_confidence = sum(confidences) / len(confidences)

        # Log individual results
        for analysis in valid_analyses:
            provider = analysis["provider"]
            confidence = analysis["confidence"]
            reasoning = (
                analysis["reasoning"][:60] + "..."
                if len(analysis["reasoning"]) > 60
                else analysis["reasoning"]
            )
            logger.info(f"   {provider}: {confidence:.1%} - {reasoning}")

        final_result = {
            "game_id": game_id,
            "game_info": game_info,
            "consensus_confidence": round(consensus_confidence, 3),
            "individual_analyses": valid_analyses,
            "providers_count": len(valid_analyses),
            "cache_status": "fresh_analysis",
        }

        logger.info(
            f"📊 Consensus for {game_id}: {consensus_confidence:.1%} from {len(valid_analyses)} providers"
        )

        # Step 5: Cache the result
        self._update_cache(cache_key, final_result)

        return final_result

    async def analyze_games(self, games_list: list[dict]) -> dict[str, Any]:
        """
        Analyze multiple games efficiently with parallel processing and caching.

        Args:
            games_list: List of game info dictionaries

        Returns:
            Dictionary mapping game_id to analysis results
        """
        if not games_list:
            logger.warning("No games provided for analysis")
            return {}

        logger.info(
            f"🎯 Analyzing {len(games_list)} games with {len(self.clients)} AI providers"
        )

        # Analyze all games in parallel
        analysis_tasks = [self.analyze_game(game) for game in games_list]
        analysis_results = await asyncio.gather(*analysis_tasks)

        # Convert to dictionary and gather stats
        results_dict = {}
        cache_hits = 0
        fresh_analyses = 0

        for result in analysis_results:
            game_id = result.get("game_id", "unknown")
            results_dict[game_id] = result

            # Track cache performance
            cache_status = result.get("cache_status", "unknown")
            if "cache" in cache_status or cache_status == "fresh_analysis":
                if cache_status == "fresh_analysis":
                    fresh_analyses += 1
                else:
                    cache_hits += 1

        # Summary statistics
        if results_dict:
            confidences = [r["consensus_confidence"] for r in results_dict.values()]
            avg_confidence = sum(confidences) / len(confidences)

            logger.info(f"✅ Analysis complete:")
            logger.info(f"   Games analyzed: {len(results_dict)}")
            logger.info(f"   Average confidence: {avg_confidence:.1%}")
            logger.info(
                f"   Cache hits: {cache_hits}, Fresh analyses: {fresh_analyses}"
            )

        return results_dict

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_entries = len(self.cache)
        current_time = datetime.now(timezone.utc)

        recent_entries = 0
        for cached_item in self.cache.values():
            try:
                timestamp = datetime.fromisoformat(cached_item["timestamp"])
                if current_time - timestamp < timedelta(hours=24):
                    recent_entries += 1
            except:
                pass

        return {
            "total_cached_games": total_entries,
            "recent_entries_24h": recent_entries,
            "cache_duration_hours": self.cache_duration.total_seconds() / 3600,
            "cache_file_size_mb": (
                os.path.getsize(self.cache_file) / 1024 / 1024
                if os.path.exists(self.cache_file)
                else 0
            ),
        }


# --- Testing and Validation ---


async def test_cached_game_selector():
    """Test the cached game selector with sample data."""
    sample_games = [
        {
            "game_id": "test_game_1",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "commence_time": "2025-06-16T19:00:00Z",
        },
        {
            "game_id": "test_game_2",
            "home_team": "Los Angeles Dodgers",
            "away_team": "San Francisco Giants",
            "commence_time": "2025-06-16T22:00:00Z",
        },
    ]

    selector = GameSelector(cache_duration_hours=1)  # Short cache for testing

    # First run - should be fresh analyses
    logger.info("🧪 First run (should be fresh analyses):")
    results1 = await selector.analyze_games(sample_games)

    # Second run - should use cache
    logger.info("\n🧪 Second run (should use cache):")
    results2 = await selector.analyze_games(sample_games)

    # Display results
    print("\n" + "=" * 60)
    print("🎯 CACHED GAME SELECTOR TEST RESULTS")
    print("=" * 60)

    for game_id, analysis in results1.items():
        print(f"\nGame: {game_id}")
        print(f"Consensus: {analysis['consensus_confidence']:.1%}")
        print(f"Cache Status: {analysis.get('cache_status', 'unknown')}")

        for individual in analysis["individual_analyses"]:
            provider = individual["provider"]
            confidence = individual["confidence"]
            reasoning = (
                individual["reasoning"][:80] + "..."
                if len(individual["reasoning"]) > 80
                else individual["reasoning"]
            )
            print(f"  {provider}: {confidence:.1%} - {reasoning}")

    # Cache statistics
    cache_stats = selector.get_cache_stats()
    print(f"\n📊 Cache Statistics:")
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_cached_game_selector())
