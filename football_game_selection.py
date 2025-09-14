#!/usr/bin/env python3
"""
Football Game Selection - AI Analysis for NFL & College Football
Adapted from MLB system for football-specific analysis and factors.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

import polars as pl

import aiohttp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class FootballGameRecommendation:
    """Structured recommendation from LLM analysis for football."""

    game_id: str
    home_team: str
    away_team: str
    confidence_score: float
    reasoning: str
    key_factors: list[str]
    risk_assessment: str
    betting_strategy: str | None = None
    expected_value: float | None = None
    market_type: str = "moneyline"  # moneyline, spread, total, prop

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "confidence_score": self.confidence_score,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "risk_assessment": self.risk_assessment,
            "betting_strategy": self.betting_strategy,
            "expected_value": self.expected_value,
            "market_type": self.market_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FootballGameRecommendation":
        required_fields = [
            "game_id",
            "home_team",
            "away_team",
            "confidence_score",
            "reasoning",
            "key_factors",
            "risk_assessment",
        ]
        for required_field in required_fields:
            if required_field not in data:
                raise ValueError(f"Missing required field: {required_field}")
        return cls(
            game_id=data["game_id"],
            home_team=data["home_team"],
            away_team=data["away_team"],
            confidence_score=float(data["confidence_score"]),
            reasoning=data["reasoning"],
            key_factors=(data["key_factors"] if isinstance(data["key_factors"], list) else []),
            risk_assessment=data["risk_assessment"],
            betting_strategy=data.get("betting_strategy"),
            expected_value=(
                float(data["expected_value"]) if data.get("expected_value") is not None else None
            ),
            market_type=data.get("market_type", "moneyline"),
        )


@dataclass
class FootballLLMAnalysisRequest:
    """Request for football game analysis."""

    games_data: pl.DataFrame
    context: dict
    max_recommendations: int = 5
    analysis_focus: str = "value_betting"
    sport_type: str = "nfl"  # nfl or ncaaf

    def to_context_string(self) -> str:
        """Converts the request to a context string for LLM analysis."""
        context_str = f"""
FOOTBALL GAME ANALYSIS REQUEST
Sport Type: {self.sport_type.upper()}
Analysis Focus: {self.analysis_focus}
Max Recommendations: {self.max_recommendations}

CONTEXT:
{json.dumps(self.context, indent=2)}

GAMES DATA:
{self.games_data.head(10).to_dicts()}
"""
        return context_str


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    async def analyze_games(
        self, request: FootballLLMAnalysisRequest
    ) -> list[FootballGameRecommendation]:
        pass


class ClaudeProvider(LLMProvider):
    """Claude AI provider for football analysis."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-latest"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
        self.timeout = 120

    @property
    def name(self) -> str:
        return "Claude"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def analyze_games(
        self, request: FootballLLMAnalysisRequest
    ) -> list[FootballGameRecommendation]:
        system_prompt = (
            "You are an expert in football betting analysis with deep knowledge of "
            "NFL and college football. Analyze games considering:\n"
            "- Quarterback matchups and recent performance\n"
            "- Offensive and defensive efficiency metrics\n"
            "- Weather conditions and their impact on passing/running\n"
            "- Home field advantage (typically 2-3 points in NFL)\n"
            "- Injury reports and lineup changes\n"
            "- Historical head-to-head performance\n"
            "- Rest days and travel fatigue\n"
            "- Coaching strategies and tendencies\n"
            "- Special teams impact\n"
            "- Recent team momentum and psychological factors\n\n"
            "Provide structured betting recommendations with confidence scores."
        )

        user_prompt = (
            f"{request.to_context_string()}\n\n"
            f"**INSTRUCTIONS:**\n"
            f"Analyze the football games and provide up to "
            f"{request.max_recommendations} recommendations. "
            f"For each recommendation:\n"
            f"- Calculate betting edge and expected value\n"
            f"- Consider football-specific factors (QB play, weather, etc.)\n"
            f"- Output in JSON format:\n"
            f"{{\n"
            f'    "recommendations": [\n'
            f"        {{\n"
            f'            "game_id": "string",\n'
            f'            "home_team": "string",\n'
            f'            "away_team": "string",\n'
            f'            "confidence_score": 0.75,\n'
            f'            "reasoning": "Football-specific analysis...",\n'
            f'            "key_factors": ["qb_matchup", "weather", "defense"],\n'
            f'            "risk_assessment": "low/medium/high with explanation",\n'
            f'            "betting_strategy": "moneyline home",\n'
            f'            "expected_value": 0.15,\n'
            f'            "market_type": "moneyline"\n'
            f"        }}\n"
            f"    ]\n"
            f"}}\n"
        )

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "max_tokens": 2000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return self._parse_response(result)
            except Exception as e:
                logger.error(f"❌ {self.name} error: {e}")
                return []

    def _parse_response(self, response: dict) -> list[FootballGameRecommendation]:
        try:
            content = response["content"][0]["text"].strip()
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                logger.error(f"❌ No JSON found in {self.name} response")
                return []
            json_str = content[start_idx:end_idx]
            json_data = json.loads(json_str)
            recommendations = []
            for rec_data in json_data.get("recommendations", []):
                try:
                    rec = FootballGameRecommendation.from_dict(rec_data)
                    if not 0.0 <= rec.confidence_score <= 1.0:
                        logger.warning(f"⚠️ Invalid confidence score {rec.confidence_score}")
                        rec.confidence_score = max(0.0, min(1.0, rec.confidence_score))
                    recommendations.append(rec)
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Skipping invalid recommendation: {e}")
                    continue
            logger.info(f"✅ {self.name} parsed {len(recommendations)} recommendations")
            return recommendations
        except Exception as e:
            logger.error(f"❌ {self.name} parsing error: {e}")
            return []




class OpenAIProvider(LLMProvider):
    """OpenAI (ChatGPT) provider for football analysis."""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        self.timeout = 120

    @property
    def name(self) -> str:
        return "ChatGPT"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def analyze_games(
        self, request: FootballLLMAnalysisRequest
    ) -> list[FootballGameRecommendation]:
        system_prompt = (
            "You are an expert in football betting analysis. Analyze games thoroughly and "
            "provide structured recommendations with confidence scores."
        )
        user_prompt = (
            f"{request.to_context_string()}\n\nReturn ONLY the JSON object described above."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}  # Force JSON response
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return self._parse_response(result)
            except Exception as e:
                logger.error(f"❌ {self.name} error: {e}")
                return []

    def _parse_response(self, response: dict) -> list[FootballGameRecommendation]:
        try:
            # Handle both regular content and function call responses
            choices = response.get("choices", [])
            if not choices:
                logger.error(f"❌ {self.name} returned no choices")
                return []
            
            message = choices[0].get("message", {})
            content = message.get("content", "").strip()
            
            # If content is already valid JSON, parse it directly
            try:
                json_data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from content
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx == -1 or end_idx == 0:
                    logger.error(f"❌ No JSON found in {self.name} response")
                    return []
                json_str = content[start_idx:end_idx]
                json_data = json.loads(json_str)
            
            recommendations = []
            for rec_data in json_data.get("recommendations", []):
                try:
                    rec = FootballGameRecommendation.from_dict(rec_data)
                    if not 0.0 <= rec.confidence_score <= 1.0:
                        logger.warning(f"⚠️ Invalid confidence score {rec.confidence_score}")
                        rec.confidence_score = max(0.0, min(1.0, rec.confidence_score))
                    recommendations.append(rec)
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Skipping invalid recommendation: {e}")
                    continue
            logger.info(f"✅ {self.name} parsed {len(recommendations)} recommendations")
            return recommendations
        except Exception as e:
            logger.error(f"❌ {self.name} parsing error: {e}")
            return []


class GrokProvider(LLMProvider):
    """Grok (xAI) provider using OpenAI-compatible chat completions API."""

    def __init__(self, api_key: str, model: str = "grok-1"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.x.ai/v1"
        self.timeout = 120

    @property
    def name(self) -> str:
        return "Grok"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def analyze_games(
        self, request: FootballLLMAnalysisRequest
    ) -> list[FootballGameRecommendation]:
        system_prompt = (
            "You are Grok-1, an expert football betting analyst. "
            "Analyze games with emphasis on QB play, offensive/defensive efficiency, "
            "weather impact, injuries, and coaching tendencies. Return concise, structured recommendations."
        )

        user_prompt = (
            f"{request.to_context_string()}\n\n"
            f"Return ONLY the JSON object described above with 'recommendations' list."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2000,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return self._parse_response(result)
            except Exception as e:
                logger.error(f"❌ {self.name} error: {e}")
                return []

    def _parse_response(self, response: dict) -> list[FootballGameRecommendation]:
        try:
            content = response["choices"][0]["message"]["content"].strip()
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                logger.error(f"❌ No JSON found in {self.name} response")
                return []
            json_str = content[start_idx:end_idx]
            json_data = json.loads(json_str)
            recs: list[FootballGameRecommendation] = []
            for rec_data in json_data.get("recommendations", []):
                try:
                    rec = FootballGameRecommendation.from_dict(rec_data)
                    if not 0.0 <= rec.confidence_score <= 1.0:
                        rec.confidence_score = max(0.0, min(1.0, rec.confidence_score))
                    recs.append(rec)
                except Exception as e:
                    logger.warning(f"⚠️ Skipping invalid recommendation from {self.name}: {e}")
            logger.info(f"✅ {self.name} parsed {len(recs)} recommendations")
            return recs
        except Exception as e:
            logger.error(f"❌ {self.name} parsing error: {e}")
            return []


class PerplexityProvider(LLMProvider):
    """Perplexity provider (OpenAI-compatible chat completions)."""

    def __init__(self, api_key: str, model: str = "llama-3.1-sonar-small-128k-online"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.perplexity.ai"
        self.timeout = 120

    @property
    def name(self) -> str:
        return "Perplexity"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def analyze_games(
        self, request: FootballLLMAnalysisRequest
    ) -> list[FootballGameRecommendation]:
        system_prompt = (
            "You are an expert football analyst focusing on betting value. Provide concise, "
            "structured recommendations with confidence and expected value."
        )
        user_prompt = (
            f"{request.to_context_string()}\n\nReturn ONLY the JSON object described above."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2000,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return self._parse_response(result)
            except Exception as e:
                logger.error(f"❌ {self.name} error: {e}")
                return []

    def _parse_response(self, response: dict) -> list[FootballGameRecommendation]:
        try:
            content = response["choices"][0]["message"]["content"].strip()
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                logger.error(f"❌ No JSON found in {self.name} response")
                return []
            json_str = content[start_idx:end_idx]
            json_data = json.loads(json_str)
            recs: list[FootballGameRecommendation] = []
            for rec_data in json_data.get("recommendations", []):
                try:
                    recs.append(FootballGameRecommendation.from_dict(rec_data))
                except Exception as e:
                    logger.warning(f"⚠️ Skipping invalid recommendation from {self.name}: {e}")
            logger.info(f"✅ {self.name} parsed {len(recs)} recommendations")
            return recs
        except Exception as e:
            logger.error(f"❌ {self.name} parsing error: {e}")
            return []


class GeminiProvider(LLMProvider):
    """Gemini provider using Google Generative Language API."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model  # Use flash model to avoid rate limits
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.timeout = 120

    @property
    def name(self) -> str:
        return "Gemini"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def analyze_games(
        self, request: FootballLLMAnalysisRequest
    ) -> list[FootballGameRecommendation]:
        system_prompt = (
            "You are an expert football betting analyst. "
            "Provide structured JSON recommendations as specified."
        )
        user_prompt = (
            f"{request.to_context_string()}\n\nReturn ONLY the JSON object described above."
        )

        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": system_prompt + "\n\n" + user_prompt}]}]}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return self._parse_response(result)
            except Exception as e:
                logger.error(f"❌ {self.name} error: {e}")
                return []

    def _parse_response(self, response: dict) -> list[FootballGameRecommendation]:
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                logger.error(f"❌ {self.name} returned no candidates")
                return []
            # Extract text
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            text = "".join(part.get("text", "") for part in parts).strip()
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                logger.error(f"❌ No JSON found in {self.name} response")
                return []
            json_str = text[start_idx:end_idx]
            json_data = json.loads(json_str)
            recs: list[FootballGameRecommendation] = []
            for rec_data in json_data.get("recommendations", []):
                try:
                    recs.append(FootballGameRecommendation.from_dict(rec_data))
                except Exception as e:
                    logger.warning(f"⚠️ Skipping invalid recommendation from {self.name}: {e}")
            logger.info(f"✅ {self.name} parsed {len(recs)} recommendations")
            return recs
        except Exception as e:
            logger.error(f"❌ {self.name} parsing error: {e}")
            return []


@dataclass
class FootballSelectionConfig:
    """Configuration for football game selection."""

    primary_provider: str = "claude"
    fallback_providers: list[str] = field(default_factory=lambda: ["chatgpt"])
    consensus_threshold: int = 2
    max_recommendations: int = 5
    min_confidence_score: float = 0.7
    enable_ensemble: bool = True
    save_analysis_history: bool = True
    analysis_focus: str = "value_betting"
    timeout_seconds: int = 180
    sport_type: str = "nfl"


class FootballGameSelector:
    """Football game selection engine with AI analysis."""

    def __init__(self, config: FootballSelectionConfig | None = None):
        self.config = config or FootballSelectionConfig()
        self.providers: dict[str, LLMProvider] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize providers
        self._initialize_providers()

        if not self.providers:
            self.logger.warning(
                "No AI providers configured. Analysis will return no recommendations."
            )
        else:
            providers_list = list(self.providers.keys())
            self.logger.info(
                f"🏈 Football Game Selector initialized with {len(providers_list)} providers"
            )

    def _initialize_providers(self):
        """Initialize AI providers."""
        from api_config import get_api_keys

        api_keys = get_api_keys()

        # Paid providers
        if api_keys.get("claude"):
            self.providers["claude"] = ClaudeProvider(api_keys["claude"])

        if api_keys.get("openai"):
            self.providers["chatgpt"] = OpenAIProvider(api_keys["openai"])

        if api_keys.get("grok"):
            self.providers["grok"] = GrokProvider(api_keys["grok"], model="grok-1")

        if api_keys.get("perplexity"):
            self.providers["perplexity"] = PerplexityProvider(api_keys["perplexity"])

        if api_keys.get("gemini"):
            self.providers["gemini"] = GeminiProvider(api_keys["gemini"])
        
        # Add free providers as fallback
        try:
            from free_ai_providers import register_free_providers
            free_providers = register_free_providers(self)
            if free_providers:
                self.logger.info(f"🆓 Added {len(free_providers)} free AI providers as fallback")
        except Exception as e:
            self.logger.warning(f"Could not load free providers: {e}")

    async def analyze_games(self, games: list[dict]) -> dict:
        """Analyze football games using AI models."""
        self.logger.info(f"🏈 Starting football game analysis for {len(games)} games")

        # If no providers, return an empty result package
        if not self.providers:
            games_df = pl.DataFrame(games)
            return self._build_result_package(games_df, {}, [], [])

        # Convert games to DataFrame
        games_df = pl.DataFrame(games)

        # Create analysis request
        request = FootballLLMAnalysisRequest(
            games_data=games_df,
            context={
                "sport_type": self.config.sport_type,
                "analysis_focus": self.config.analysis_focus,
                "timestamp": datetime.now().isoformat(),
            },
            max_recommendations=self.config.max_recommendations,
            analysis_focus=self.config.analysis_focus,
            sport_type=self.config.sport_type,
        )

        # Run provider analysis
        provider_results = await self._run_provider_analysis(request)

        # Combine recommendations
        final_recommendations = self._combine_recommendations(provider_results)

        # Filter by confidence threshold
        filtered_recs = [
            rec
            for rec in final_recommendations
            if rec.confidence_score >= self.config.min_confidence_score
        ]

        # Build result package
        result_package = self._build_result_package(
            games_df, provider_results, final_recommendations, filtered_recs
        )

        self.logger.info(f"🏈 Football analysis complete: {len(filtered_recs)} recommendations")
        return result_package

    async def _run_provider_analysis(
        self, request: FootballLLMAnalysisRequest
    ) -> dict[str, list[FootballGameRecommendation]]:
        """Run analysis with multiple providers."""
        tasks = {
            name: asyncio.create_task(
                self._safe_provider_call(provider, request), name=f"analysis_{name}"
            )
            for name, provider in self.providers.items()
        }

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks.values(), return_exceptions=True),
                timeout=self.config.timeout_seconds,
            )
        except TimeoutError:
            self.logger.error(f"🏈 Analysis timed out after {self.config.timeout_seconds}s")
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            results = [[] for _ in tasks]

        provider_results: dict[str, list[FootballGameRecommendation]] = {}
        for i, (provider_name, task) in enumerate(tasks.items()):
            result = results[i]
            if isinstance(result, Exception):
                self.logger.error(f"🏈 Provider '{provider_name}' failed: {result}")
                provider_results[provider_name] = []
            else:
                recs = cast(list[FootballGameRecommendation], result)
                provider_results[provider_name] = recs
                self.logger.info(f"🏈 {provider_name}: {len(recs)} recommendations")

        return provider_results

    async def _safe_provider_call(
        self, provider: LLMProvider, request: FootballLLMAnalysisRequest
    ) -> list[FootballGameRecommendation]:
        """Safely call a provider with error handling."""
        try:
            return await provider.analyze_games(request)
        except Exception as e:
            self.logger.error(f"🏈 Provider {provider.name} error: {e}")
            return []

    def _combine_recommendations(
        self, provider_results: dict[str, list[FootballGameRecommendation]]
    ) -> list[FootballGameRecommendation]:
        """Combine recommendations from multiple providers."""
        if not provider_results:
            return []

        if not self.config.enable_ensemble or len(provider_results) == 1:
            primary_results = provider_results.get(self.config.primary_provider, [])
            if primary_results:
                return primary_results
            for results in provider_results.values():
                if results:
                    return results
            return []

        # Ensemble logic for multiple providers
        @dataclass
        class ConsensusBucket:
            recommendations: list[FootballGameRecommendation]
            total_confidence: float
            provider_count: int
            sample_rec: FootballGameRecommendation

        game_consensus: dict[str, ConsensusBucket] = {}
        for provider_name, recommendations in provider_results.items():
            for rec in recommendations:
                game_key = f"{rec.away_team}_at_{rec.home_team}".lower()
                if game_key not in game_consensus:
                    game_consensus[game_key] = ConsensusBucket(
                        recommendations=[], total_confidence=0.0, provider_count=0, sample_rec=rec
                    )
                bucket = game_consensus[game_key]
                bucket.recommendations.append(rec)
                bucket.total_confidence += rec.confidence_score
                bucket.provider_count += 1

        final_recommendations: list[FootballGameRecommendation] = []
        for game_key, consensus in game_consensus.items():
            if consensus.provider_count >= self.config.consensus_threshold:
                avg_confidence = consensus.total_confidence / consensus.provider_count
                final_rec = consensus.sample_rec
                final_rec.confidence_score = avg_confidence
                final_rec.reasoning += f" (Consensus from {consensus.provider_count} providers)"
                final_recommendations.append(final_rec)

        final_recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        if not final_recommendations:
            # Fallback: if consensus yields nothing, return the largest non-empty provider list
            for preferred in [self.config.primary_provider, *provider_results.keys()]:
                if preferred in provider_results and provider_results[preferred]:
                    return provider_results[preferred][: self.config.max_recommendations]
        return final_recommendations[: self.config.max_recommendations]

    def _build_result_package(
        self,
        games_df: pl.DataFrame,
        provider_results: dict[str, list[FootballGameRecommendation]],
        final_recs: list[FootballGameRecommendation],
        filtered_recs: list[FootballGameRecommendation],
    ) -> dict:
        """Build the final result package."""
        return {
            "recommendations": [rec.to_dict() for rec in filtered_recs],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "sport_type": self.config.sport_type,
                "analysis_focus": self.config.analysis_focus,
                "providers_used": list(provider_results.keys()),
                "providers_successful": [name for name, recs in provider_results.items() if recs],
                "total_games_analyzed": games_df.height,
                "total_features": games_df.width,
                "recommendations_generated": len(final_recs),
                "recommendations_after_filtering": len(filtered_recs),
                "confidence_threshold": self.config.min_confidence_score,
                "consensus_threshold": self.config.consensus_threshold,
                "ensemble_enabled": self.config.enable_ensemble,
            },
            "provider_analysis": {
                name: [rec.to_dict() for rec in recs] for name, recs in provider_results.items()
            },
            "analysis_summary": {
                "avg_confidence": (
                    sum(rec.confidence_score for rec in filtered_recs) / len(filtered_recs)
                    if filtered_recs
                    else 0
                ),
                "risk_distribution": self._calculate_risk_distribution(filtered_recs),
                "market_type_distribution": self._calculate_market_distribution(filtered_recs),
            },
        }

    def _calculate_risk_distribution(
        self, recommendations: list[FootballGameRecommendation]
    ) -> dict:
        """Calculate risk distribution of recommendations."""
        risk_counts = {"low": 0, "medium": 0, "high": 0}
        for rec in recommendations:
            risk = rec.risk_assessment.lower()
            if "low" in risk:
                risk_counts["low"] += 1
            elif "high" in risk:
                risk_counts["high"] += 1
            else:
                risk_counts["medium"] += 1
        return risk_counts

    def _calculate_market_distribution(
        self, recommendations: list[FootballGameRecommendation]
    ) -> dict:
        """Calculate market type distribution of recommendations."""
        market_counts = {"moneyline": 0, "spread": 0, "total": 0, "prop": 0}
        for rec in recommendations:
            market = rec.market_type.lower()
            if "moneyline" in market:
                market_counts["moneyline"] += 1
            elif "spread" in market:
                market_counts["spread"] += 1
            elif "total" in market:
                market_counts["total"] += 1
            else:
                market_counts["prop"] += 1
        return market_counts


async def main():
    """Test the football game selector."""
    from api_config import get_api_keys

    api_keys = get_api_keys()
    if not api_keys.get("claude") and not api_keys.get("openai"):
        logger.error("No AI API keys found")
        return

    # Sample football games
    sample_games = [
        {
            "game_id": "NE_BUF_20250101",
            "home_team": "Buffalo Bills",
            "away_team": "New England Patriots",
            "home_odds": 1.85,
            "away_odds": 2.05,
            "spread": -3.5,
            "total": 42.5,
        },
        {
            "game_id": "KC_LV_20250101",
            "home_team": "Las Vegas Raiders",
            "away_team": "Kansas City Chiefs",
            "home_odds": 2.50,
            "away_odds": 1.60,
            "spread": 6.5,
            "total": 48.5,
        },
    ]

    config = FootballSelectionConfig(sport_type="nfl")
    selector = FootballGameSelector(config)

    results = await selector.analyze_games(sample_games)
    logger.info(f"🏈 Analysis results: {len(results.get('recommendations', []))} recommendations")


if __name__ == "__main__":
    asyncio.run(main())
