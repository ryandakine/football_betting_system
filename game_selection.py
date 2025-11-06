import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

import aiohttp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GameRecommendation:
    """Structured recommendation from LLM analysis."""

    game_id: str
    home_team: str
    away_team: str
    confidence_score: float
    reasoning: str
    key_factors: list[str]
    risk_assessment: str
    betting_strategy: str | None = None
    expected_value: float | None = None

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
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GameRecommendation":
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
            key_factors=(
                data["key_factors"] if isinstance(data["key_factors"], list) else []
            ),
            risk_assessment=data["risk_assessment"],
            betting_strategy=data.get("betting_strategy"),
            expected_value=(
                float(data["expected_value"])
                if data.get("expected_value") is not None
                else None
            ),
        )


@dataclass
class LLMAnalysisRequest:
    games_data: pl.DataFrame
    context: dict[str, Any]
    max_recommendations: int = 5
    analysis_focus: str = "value_betting"

    def to_context_string(self) -> str:
        def format_dict(d: dict | None, fallback: str = "Not provided") -> str:
            return json.dumps(d, indent=2) if d else fallback

        priority_cols = [
            "game_id",
            "home_team",
            "away_team",
            "date",
            "home_team_win_probability",
            "home_avg_runs_scored",
            "away_avg_runs_scored",
            "weather_run_suppression_factor",
            "runs_impact_per_game",
            "home_recent_win_rate",
            "away_recent_win_rate",
        ]
        available_cols = [
            col for col in priority_cols if col in self.games_data.columns
        ]
        if not available_cols:
            available_cols = self.games_data.columns[:15]

        games_summary = (
            (
                self.games_data.select(available_cols)
                .head(20)
                .to_pandas()
                .to_string(index=False)
            )
            if self.games_data.height > 0 and available_cols
            else "No game data available"
        )

        return (
            f"MLB BETTING ANALYSIS REQUEST - "
            f"{datetime.now().strftime('%Y-%m-%d')}**\n\n"
            f"**Analysis Focus:** {self.analysis_focus}\n"
            f"**Maximum Recommendations Requested:** {self.max_recommendations}\n\n"
            f"**AVAILABLE GAMES DATA:**\n{games_summary}\n\n"
            f"**HISTORICAL PERFORMANCE CONTEXT:**\n"
            f"{format_dict(self.context.get('historical_performance'))}\n\n"
            f"**CURRENT MARKET CONDITIONS:**\n"
            f"{format_dict(self.context.get('market_conditions'))}\n\n"
            f"**BETTING CONSTRAINTS & STRATEGY:**\n"
            f"{format_dict(self.context.get('constraints'))}\n\n"
            f"**ADDITIONAL CONTEXT:**\n"
            f"{format_dict(self.context.get('additional_info'))}"
        )


class LLMProvider(ABC):
    @abstractmethod
    async def analyze_games(
        self, request: LLMAnalysisRequest
    ) -> list[GameRecommendation]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class GrokProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "grok-3"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.x.ai/v1"
        self.timeout = 120

    @property
    def name(self) -> str:
        return "Grok"

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key != "fake-key")

    async def analyze_games(
        self, request: LLMAnalysisRequest
    ) -> list[GameRecommendation]:
        system_prompt = (
            "You are an expert in mathematical reasoning and MLB betting analysis. "
            "Use your 'Think' mode to calculate betting edges and expected values "
            "using the Kelly Criterion. Focus on:\n"
            "- Quantitative analysis of win probabilities and odds\n"
            "- Identifying value bets based on edge calculations\n"
            "- Providing precise numerical reasoning\n"
        )
        user_prompt = (
            f"{request.to_context_string()}\n\n"
            f"**INSTRUCTIONS:**\n"
            f"Analyze the games and provide up to "
            f"{request.max_recommendations} recommendations. For each:\n"
            f"- Calculate the betting edge (model probability - implied probability)\n"
            f"- Compute expected value using the Kelly Criterion\n"
            f"- Output in JSON format:\n"
            f"{{\n"
            f'    "recommendations": [\n'
            f"        {{\n"
            f'            "game_id": "string",\n'
            f'            "home_team": "string",\n'
            f'            "away_team": "string",\n'
            f'            "confidence_score": 0.75,\n'
            f'            "reasoning": "Quantitative analysis...",\n'
            f'            "key_factors": ["edge", "odds", "probability"],\n'
            f'            "risk_assessment": "low/medium/high with explanation",\n'
            f'            "betting_strategy": "moneyline home",\n'
            f'            "expected_value": 0.15\n'
            f"        }}\n"
            f"    ]\n"
            f"}}\n"
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

    def _parse_response(self, response: dict) -> list[GameRecommendation]:
        try:
            content = response["choices"][0]["message"]["content"].strip()
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
                    rec = GameRecommendation.from_dict(rec_data)
                    if not 0.0 <= rec.confidence_score <= 1.0:
                        logger.warning(
                            f"⚠️ Invalid confidence score {rec.confidence_score}"
                        )
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
    def __init__(self, api_key: str, model: str = "gpt-4o"):
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
        self, request: LLMAnalysisRequest
    ) -> list[GameRecommendation]:
        system_prompt = (
            "You are Codex, an expert in code generation and MLB betting analysis. "
            "Generate Python code to analyze betting opportunities, focusing on:\n"
            "- Edge calculations (model probability - implied probability)\n"
            "- Narrative explanations for recommendations\n"
            "- Structured JSON output\n"
        )
        user_prompt = (
            f"{request.to_context_string()}\n\n"
            f"**INSTRUCTIONS:**\n"
            f"Write Python code to analyze up to {request.max_recommendations} games. "
            f"Return recommendations in JSON format:\n"
            f"{{\n"
            f'    "recommendations": [\n'
            f"        {{\n"
            f'            "game_id": "string",\n'
            f'            "home_team": "string",\n'
            f'            "away_team": "string",\n'
            f'            "confidence_score": 0.75,\n'
            f'            "reasoning": "Code-driven analysis...",\n'
            f'            "key_factors": ["edge", "code", "narrative"],\n'
            f'            "risk_assessment": "low/medium/high with explanation",\n'
            f'            "betting_strategy": "moneyline home",\n'
            f'            "expected_value": 0.15\n'
            f"        }}\n"
            f"    ]\n"
            f"}}\n"
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

    def _parse_response(self, response: dict) -> list[GameRecommendation]:
        try:
            content = response["choices"][0]["message"]["content"].strip()
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
                    rec = GameRecommendation.from_dict(rec_data)
                    if not 0.0 <= rec.confidence_score <= 1.0:
                        logger.warning(
                            f"⚠️ Invalid confidence score {rec.confidence_score}"
                        )
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


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3.5-sonnet"):
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
        self, request: LLMAnalysisRequest
    ) -> list[GameRecommendation]:
        system_prompt = (
            "You are an expert in MLB betting with superior reasoning and coding "
            "skills. Generate structured betting recommendations, focusing on:\n"
            "- Strategic frameworks for value betting\n"
            "- Detailed reasoning with logical steps\n"
            "- Robust risk assessment\n"
        )
        user_prompt = (
            f"{request.to_context_string()}\n\n"
            f"**INSTRUCTIONS:**\n"
            f"Provide up to {request.max_recommendations} recommendations with "
            f"structured reasoning. Output in JSON format:\n"
            f"{{\n"
            f'    "recommendations": [\n'
            f"        {{\n"
            f'            "game_id": "string",\n'
            f'            "home_team": "string",\n'
            f'            "away_team": "string",\n'
            f'            "confidence_score": 0.75,\n'
            f'            "reasoning": "Structured reasoning...",\n'
            f'            "key_factors": ["strategy", "risk", "logic"],\n'
            f'            "risk_assessment": "low/medium/high with explanation",\n'
            f'            "betting_strategy": "moneyline home",\n'
            f'            "expected_value": 0.15\n'
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
            "messages": [{"role": "user", "content": user_prompt}],
            "system": system_prompt,
            "max_tokens": 2000,
            "temperature": 0.1,
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

    def _parse_response(self, response: dict) -> list[GameRecommendation]:
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
                    rec = GameRecommendation.from_dict(rec_data)
                    if not 0.0 <= rec.confidence_score <= 1.0:
                        logger.warning(
                            f"⚠️ Invalid confidence score {rec.confidence_score}"
                        )
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


class MockGrokProvider(GrokProvider):
    def __init__(self, api_key: str = "mock-key"):
        super().__init__(api_key)

    @property
    def name(self) -> str:
        return "MockGrok"

    def is_available(self) -> bool:
        return True

    async def analyze_games(
        self, request: LLMAnalysisRequest
    ) -> list[GameRecommendation]:
        await asyncio.sleep(0.1)
        recommendations = []
        if request.games_data.height > 0:
            games_sample = request.games_data.head(3).to_dicts()
            for i, game in enumerate(games_sample):
                rec = GameRecommendation(
                    game_id=game.get("game_id", f"mock_game_{i}"),
                    home_team=game.get("home_team", f"HOME_{i}"),
                    away_team=game.get("away_team", f"AWAY_{i}"),
                    confidence_score=0.75 + (i * 0.05),
                    reasoning=f"Mock analysis: Strong edge for game {i+1}.",
                    key_factors=["pitcher_matchup", "recent_form", "weather"],
                    risk_assessment="medium - standard variance",
                    betting_strategy=f"Bet {game.get('home_team', 'HOME')} moneyline",
                    expected_value=0.12 + (i * 0.03),
                )
                recommendations.append(rec)
        logger.info(f"MockGrok generated {len(recommendations)} mock recommendations")
        return recommendations


@dataclass
class SelectionConfig:
    primary_provider: str = "claude"
    fallback_providers: list[str] = field(default_factory=lambda: ["grok", "chatgpt"])
    consensus_threshold: int = 2
    max_recommendations: int = 5
    min_confidence_score: float = 0.7
    enable_ensemble: bool = True
    save_analysis_history: bool = True
    analysis_focus: str = "value_betting"
    timeout_seconds: int = 180


class GameSelectionEngine:
    def __init__(self, config: SelectionConfig, providers: dict[str, LLMProvider]):
        self.config = config
        self.providers = {
            name: p for name, p in providers.items() if p and p.is_available()
        }
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.providers:
            raise ValueError("No available LLM providers configured")
        self.logger.info(f"Initialized with providers: {list(self.providers.keys())}")

    async def select_games(self, processed_data: pl.DataFrame, context: dict) -> dict:
        self.logger.info("Starting LLM game selection analysis")
        self.logger.info(
            f"Input: {processed_data.height} games, {processed_data.width} features"
        )
        request = LLMAnalysisRequest(
            games_data=processed_data,
            context=context,
            max_recommendations=self.config.max_recommendations,
            analysis_focus=self.config.analysis_focus,
        )
        provider_results = await self._run_provider_analysis(request)
        final_recommendations = self._combine_recommendations(provider_results)
        filtered_recs = [
            rec
            for rec in final_recommendations
            if rec.confidence_score >= self.config.min_confidence_score
        ]
        result_package = self._build_result_package(
            processed_data, provider_results, final_recommendations, filtered_recs
        )
        if self.config.save_analysis_history:
            await self._save_analysis_history(result_package, processed_data)
        self.logger.info(f"Analysis complete: {len(filtered_recs)} recommendations")
        return result_package

    async def _run_provider_analysis(
        self, request: LLMAnalysisRequest
    ) -> dict[str, list[GameRecommendation]]:
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
            self.logger.error(
                f"Analysis timed out after {self.config.timeout_seconds}s"
            )
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            results = [[] for _ in tasks]
        provider_results = {}
        for i, (provider_name, task) in enumerate(tasks.items()):
            result = results[i]
            if isinstance(result, Exception):
                self.logger.error(f"Provider '{provider_name}' failed: {result}")
                provider_results[provider_name] = []
            else:
                provider_results[provider_name] = result
                self.logger.info(f"{provider_name}: {len(result)} recommendations")
        return provider_results

    async def _safe_provider_call(
        self, provider: LLMProvider, request: LLMAnalysisRequest
    ) -> list[GameRecommendation]:
        try:
            return await provider.analyze_games(request)
        except Exception as e:
            self.logger.error(f"Provider {provider.name} error: {e}")
            return []

    def _combine_recommendations(
        self, provider_results: dict[str, list[GameRecommendation]]
    ) -> list[GameRecommendation]:
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
        game_consensus = {}
        for provider_name, recommendations in provider_results.items():
            for rec in recommendations:
                game_key = f"{rec.away_team}_at_{rec.home_team}".lower()
                if game_key not in game_consensus:
                    game_consensus[game_key] = {
                        "recommendations": [],
                        "total_confidence": 0,
                        "provider_count": 0,
                        "sample_rec": rec,
                    }
                game_consensus[game_key]["recommendations"].append(rec)
                game_consensus[game_key]["total_confidence"] += rec.confidence_score
                game_consensus[game_key]["provider_count"] += 1
        final_recommendations = []
        for game_key, consensus in game_consensus.items():
            if consensus["provider_count"] >= self.config.consensus_threshold:
                avg_confidence = (
                    consensus["total_confidence"] / consensus["provider_count"]
                )
                final_rec = consensus["sample_rec"]
                final_rec.confidence_score = avg_confidence
                final_rec.reasoning += (
                    f" (Consensus from {consensus['provider_count']} providers)"
                )
                final_recommendations.append(final_rec)
        final_recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        return final_recommendations[: self.config.max_recommendations]

    def _build_result_package(
        self,
        processed_data: pl.DataFrame,
        provider_results: dict[str, list[GameRecommendation]],
        final_recs: list[GameRecommendation],
        filtered_recs: list[GameRecommendation],
    ) -> dict:
        return {
            "recommendations": [rec.to_dict() for rec in filtered_recs],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "analysis_focus": self.config.analysis_focus,
                "providers_used": list(provider_results.keys()),
                "providers_successful": [
                    name for name, recs in provider_results.items() if recs
                ],
                "total_games_analyzed": processed_data.height,
                "total_features": processed_data.width,
                "recommendations_generated": len(final_recs),
                "recommendations_after_filtering": len(filtered_recs),
                "confidence_threshold": self.config.min_confidence_score,
                "consensus_threshold": self.config.consensus_threshold,
                "ensemble_enabled": self.config.enable_ensemble,
            },
            "provider_analysis": {
                name: [rec.to_dict() for rec in recs]
                for name, recs in provider_results.items()
            },
            "analysis_summary": {
                "avg_confidence": (
                    sum(rec.confidence_score for rec in filtered_recs)
                    / len(filtered_recs)
                    if filtered_recs
                    else 0
                ),
                "risk_distribution": self._calculate_risk_distribution(filtered_recs),
                "strategy_distribution": self._calculate_strategy_distribution(
                    filtered_recs
                ),
            },
        }

    def _calculate_risk_distribution(
        self, recommendations: list[GameRecommendation]
    ) -> dict[str, int]:
        risk_counts = {"low": 0, "medium": 0, "high": 0, "other": 0}
        for rec in recommendations:
            risk_level = rec.risk_assessment.lower()
            if "low" in risk_level:
                risk_counts["low"] += 1
            elif "medium" in risk_level:
                risk_counts["medium"] += 1
            elif "high" in risk_level:
                risk_counts["high"] += 1
            else:
                risk_counts["other"] += 1
        return risk_counts

    def _calculate_strategy_distribution(
        self, recommendations: list[GameRecommendation]
    ) -> dict[str, int]:
        strategy_counts = {}
        for rec in recommendations:
            if rec.betting_strategy:
                strategy_type = rec.betting_strategy.lower()
                if "moneyline" in strategy_type:
                    key = "moneyline"
                elif "spread" in strategy_type or "run line" in strategy_type:
                    key = "spread"
                elif (
                    "total" in strategy_type
                    or "over" in strategy_type
                    or "under" in strategy_type
                ):
                    key = "totals"
                else:
                    key = "other"
                strategy_counts[key] = strategy_counts.get(key, 0) + 1
        return strategy_counts

    async def _save_analysis_history(self, result: dict, processed_data: pl.DataFrame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_dir = Path("analysis_history")
        history_dir.mkdir(exist_ok=True)
        try:
            history_file = history_dir / f"analysis_{timestamp}.json"
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)
            data_file = history_dir / f"data_{timestamp}.parquet"
            processed_data.write_parquet(data_file)
            self.logger.info(f"Analysis history saved: {history_file}")
        except Exception as e:
            self.logger.error(f"Failed to save analysis history: {e}")


async def run_complete_pipeline(
    selection_config: SelectionConfig,
    llm_providers: dict[str, LLMProvider],
    additional_context: dict,
    processed_data: pl.DataFrame | None = None,
) -> dict:
    logger.info("Starting ML pipeline: Data Processing → LLM Game Selection")
    if processed_data is None:
        processed_data = pl.DataFrame(
            {
                "game_id": [f"GAME_{i:03d}_20250610" for i in range(1, 24)],
                "home_team": [
                    "NYY",
                    "LAD",
                    "HOU",
                    "CHC",
                    "ATL",
                    "PHI",
                    "TOR",
                    "DET",
                    "MIN",
                    "MIL",
                    "CIN",
                    "OAK",
                    "SD",
                    "COL",
                    "BAL",
                    "BOS",
                    "SF",
                    "TEX",
                    "STL",
                    "MIA",
                    "NYM",
                    "TB",
                    "KC",
                ],
                "away_team": [
                    "BOS",
                    "SF",
                    "TEX",
                    "STL",
                    "MIA",
                    "NYM",
                    "TB",
                    "CLE",
                    "KC",
                    "PIT",
                    "CHW",
                    "SEA",
                    "ARI",
                    "LAA",
                    "WSH",
                    "NYY",
                    "LAD",
                    "HOU",
                    "CHC",
                    "ATL",
                    "PHI",
                    "TOR",
                    "MIN",
                ],
                "date": ["2025-06-10"] * 23,
                "home_team_win_probability": [0.55 + i * 0.01 for i in range(23)],
                "home_avg_runs_scored": [4.5 + i * 0.1 for i in range(23)],
                "away_avg_runs_scored": [4.2 + i * 0.1 for i in range(23)],
                "weather_run_suppression_factor": [0.1] * 23,
                "runs_impact_per_game": [0.05] * 23,
                "home_recent_win_rate": [0.6] * 23,
                "away_recent_win_rate": [0.55] * 23,
            }
        )
        logger.info(f"Using sample data: {processed_data.height} games")
    else:
        logger.info(
            f"Using provided data: {processed_data.height} games, "
            f"{processed_data.width} features"
        )
    try:
        engine = GameSelectionEngine(selection_config, llm_providers)
        selection_results = await engine.select_games(
            processed_data, additional_context
        )
        _display_results(processed_data, selection_results)
        return selection_results
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def _display_results(processed_data: pl.DataFrame, results: dict):
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    metadata = results["metadata"]
    print(f"Games Processed: {metadata['total_games_analyzed']}")
    print(f"Features Available: {metadata['total_features']}")
    print(f"Providers Used: {', '.join(metadata['providers_successful'])}")
    print(f"Recommendations Generated: {metadata['recommendations_generated']}")
    print(f"Final Recommendations: {metadata['recommendations_after_filtering']}")
    print(f"Avg Confidence: {results['analysis_summary']['avg_confidence']:.1%}")
    if results["recommendations"]:
        print("\nTOP RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(results["recommendations"][:5], 1):
            print(f"\n{i}. {rec['away_team']} @ {rec['home_team']}")
            print(f"   Confidence: {rec['confidence_score']:.1%}")
            print(f"   Strategy: {rec['betting_strategy']}")
            print(f"   Expected Value: {rec.get('expected_value', 'N/A')}")
            print(f"   Risk: {rec['risk_assessment']}")
            print(f"   Key Factors: {', '.join(rec['key_factors'][:3])}")
            print(f"   Reasoning: {rec['reasoning'][:120]}...")
    else:
        print("\nNo recommendations met the confidence threshold")
    risk_dist = results["analysis_summary"]["risk_distribution"]
    strategy_dist = results["analysis_summary"]["strategy_distribution"]
    if any(risk_dist.values()):
        print(f"\nRisk Distribution: {dict(risk_dist)}")
    if any(strategy_dist.values()):
        print(f"\nStrategy Distribution: {dict(strategy_dist)}")


async def main():
    providers = {
        "grok": MockGrokProvider(),
        "chatgpt": OpenAIProvider(os.getenv("OPENAI_API_KEY", "fake-openai-key")),
        "claude": ClaudeProvider(os.getenv("ANTHROPIC_API_KEY", "fake-claude-key")),
    }
    selection_config = SelectionConfig(
        primary_provider="claude",
        max_recommendations=5,
        min_confidence_score=0.70,
        enable_ensemble=True,
        save_analysis_history=True,
        analysis_focus="value_betting",
    )
    additional_context = {
        "historical_performance": {
            "last_30_days_roi": 0.125,
            "win_rate": 0.647,
            "profitable_strategies": ["home_favorites", "under_totals"],
            "losing_patterns": ["away_underdogs", "high_total_overs"],
        },
        "market_conditions": {
            "season_stage": "mid_season",
            "recent_trends": "pitching_dominant",
            "weather_factor": "moderate_impact",
            "public_sentiment": "favoring_overs",
        },
        "constraints": {"max_risk_per_game": 0.05, "max_bet_percentage": 0.1},
    }
    await run_complete_pipeline(selection_config, providers, additional_context)


if __name__ == "__main__":
    asyncio.run(main())
