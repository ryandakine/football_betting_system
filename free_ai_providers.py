#!/usr/bin/env python3
"""
Free AI Providers for Football Betting Analysis
Uses HuggingFace Inference API, Cohere, and other free/open models
"""

import asyncio
import json
import logging
import aiohttp
from typing import List, Optional
from football_game_selection import LLMProvider, FootballGameRecommendation, FootballLLMAnalysisRequest

logger = logging.getLogger(__name__)


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API provider (free tier available)"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.api_key = api_key  # Optional for public models
        self.model = model
        self.base_url = "https://api-inference.huggingface.co/models"
        self.timeout = 120
        
    @property
    def name(self) -> str:
        return "HuggingFace"
    
    def is_available(self) -> bool:
        return True  # Works without API key for public models
        
    async def analyze_games(
        self, request: FootballLLMAnalysisRequest
    ) -> list[FootballGameRecommendation]:
        prompt = f"""You are a football betting analyst. Analyze these games and provide recommendations.

{request.to_context_string()}

Return a JSON object with a 'recommendations' array. Each recommendation should have:
- game_id, home_team, away_team
- confidence_score (0.0 to 1.0)
- reasoning (brief analysis)
- key_factors (list of important factors)
- risk_assessment (low/medium/high)
- expected_value (numeric)
- market_type (moneyline/spread/total)

Limit to {request.max_recommendations} recommendations."""

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1500,
                "temperature": 0.1,
                "return_full_text": False
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}/{self.model}"
                async with session.post(
                    url, 
                    headers=headers, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 503:
                        logger.warning(f"⚠️ {self.name} model loading, retrying...")
                        await asyncio.sleep(20)
                        return []
                    
                    response.raise_for_status()
                    result = await response.json()
                    return self._parse_response(result)
            except Exception as e:
                logger.error(f"❌ {self.name} error: {e}")
                return []
    
    def _parse_response(self, response) -> list[FootballGameRecommendation]:
        try:
            # HuggingFace returns array with generated text
            if isinstance(response, list) and response:
                text = response[0].get("generated_text", "")
            else:
                text = str(response)
                
            # Extract JSON from response
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                logger.error(f"❌ No JSON found in {self.name} response")
                return []
                
            json_str = text[start_idx:end_idx]
            json_data = json.loads(json_str)
            
            recommendations = []
            for rec_data in json_data.get("recommendations", []):
                try:
                    rec = FootballGameRecommendation.from_dict(rec_data)
                    recommendations.append(rec)
                except Exception as e:
                    logger.warning(f"⚠️ Skipping invalid recommendation: {e}")
                    
            logger.info(f"✅ {self.name} parsed {len(recommendations)} recommendations")
            return recommendations
        except Exception as e:
            logger.error(f"❌ {self.name} parsing error: {e}")
            return []


class CohereProvider(LLMProvider):
    """Cohere AI provider (free tier with 1000 calls/month)"""
    
    def __init__(self, api_key: str, model: str = "command"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.cohere.ai/v1"
        self.timeout = 120
        
    @property
    def name(self) -> str:
        return "Cohere"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
        
    async def analyze_games(
        self, request: FootballLLMAnalysisRequest
    ) -> list[FootballGameRecommendation]:
        prompt = f"""Analyze these football games for betting opportunities.

{request.to_context_string()}

Provide {request.max_recommendations} recommendations in JSON format with these fields:
- game_id, home_team, away_team
- confidence_score (0.0-1.0)
- reasoning
- key_factors (array)
- risk_assessment
- expected_value
- market_type

Focus on value betting and statistical edges."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 1500,
            "temperature": 0.1
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/generate",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return self._parse_response(result)
            except Exception as e:
                logger.error(f"❌ {self.name} error: {e}")
                return []
    
    def _parse_response(self, response: dict) -> list[FootballGameRecommendation]:
        try:
            text = response.get("generations", [{}])[0].get("text", "")
            
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                logger.error(f"❌ No JSON found in {self.name} response")
                return []
                
            json_str = text[start_idx:end_idx]
            json_data = json.loads(json_str)
            
            recommendations = []
            for rec_data in json_data.get("recommendations", []):
                try:
                    rec = FootballGameRecommendation.from_dict(rec_data)
                    recommendations.append(rec)
                except Exception as e:
                    logger.warning(f"⚠️ Skipping invalid recommendation: {e}")
                    
            logger.info(f"✅ {self.name} parsed {len(recommendations)} recommendations")
            return recommendations
        except Exception as e:
            logger.error(f"❌ {self.name} parsing error: {e}")
            return []


class LocalFallbackProvider(LLMProvider):
    """Local rule-based fallback when no AI APIs are available"""
    
    def __init__(self):
        pass
        
    @property
    def name(self) -> str:
        return "LocalFallback"
    
    def is_available(self) -> bool:
        return True  # Always available
        
    async def analyze_games(
        self, request: FootballLLMAnalysisRequest
    ) -> list[FootballGameRecommendation]:
        """Simple rule-based analysis as fallback"""
        recommendations = []
        
        try:
            games_data = request.games_data.to_dicts()
            
            for game in games_data[:request.max_recommendations]:
                # Simple edge detection based on odds
                home_odds = game.get("home_odds", 2.0)
                away_odds = game.get("away_odds", 2.0)
                
                # Favor underdogs with value
                if home_odds > 2.5:
                    confidence = min(0.7, home_odds / 5.0)
                    recommendation = FootballGameRecommendation(
                        game_id=game.get("game_id", "unknown"),
                        home_team=game.get("home_team", "Home"),
                        away_team=game.get("away_team", "Away"),
                        confidence_score=confidence,
                        reasoning="Home underdog with potential value",
                        key_factors=["underdog_value", "home_advantage"],
                        risk_assessment="medium",
                        expected_value=5.0,
                        market_type="moneyline"
                    )
                    recommendations.append(recommendation)
                elif away_odds > 2.8:
                    confidence = min(0.65, away_odds / 5.5)
                    recommendation = FootballGameRecommendation(
                        game_id=game.get("game_id", "unknown"),
                        home_team=game.get("home_team", "Home"),
                        away_team=game.get("away_team", "Away"),
                        confidence_score=confidence,
                        reasoning="Away team value play",
                        key_factors=["away_value", "odds_discrepancy"],
                        risk_assessment="high",
                        expected_value=4.5,
                        market_type="moneyline"
                    )
                    recommendations.append(recommendation)
                    
            logger.info(f"✅ {self.name} generated {len(recommendations)} rule-based recommendations")
            
        except Exception as e:
            logger.error(f"❌ {self.name} error: {e}")
            
        return recommendations


def register_free_providers(selector):
    """Register free AI providers with the game selector"""
    from api_config import get_api_keys
    
    api_keys = get_api_keys()
    providers_added = []
    
    # Add HuggingFace (works without API key for public models)
    if "huggingface" not in selector.providers:
        hf_key = api_keys.get("huggingface")  # Optional
        selector.providers["huggingface"] = HuggingFaceProvider(api_key=hf_key)
        providers_added.append("huggingface")
        
    # Add Cohere if API key exists
    cohere_key = api_keys.get("cohere")
    if cohere_key and "cohere" not in selector.providers:
        selector.providers["cohere"] = CohereProvider(api_key=cohere_key)
        providers_added.append("cohere")
        
    # Always add local fallback
    if "local" not in selector.providers:
        selector.providers["local"] = LocalFallbackProvider()
        providers_added.append("local")
        
    if providers_added:
        logger.info(f"✅ Added free providers: {providers_added}")
        
    return providers_added


# Test free providers
async def test_free_providers():
    """Test the free AI providers"""
    import polars as pl
    from football_game_selection import FootballLLMAnalysisRequest
    
    # Sample game data
    games_data = pl.DataFrame([
        {
            "game_id": "TEST_001",
            "home_team": "Buffalo Bills", 
            "away_team": "Miami Dolphins",
            "home_odds": 1.85,
            "away_odds": 2.10,
            "spread": -3.5,
            "total": 48.5
        }
    ])
    
    request = FootballLLMAnalysisRequest(
        games_data=games_data,
        context={"test": True},
        max_recommendations=1,
        sport_type="nfl"
    )
    
    # Test each provider
    providers = [
        HuggingFaceProvider(),
        LocalFallbackProvider()
    ]
    
    for provider in providers:
        print(f"\nTesting {provider.name}...")
        if provider.is_available():
            recommendations = await provider.analyze_games(request)
            print(f"  Got {len(recommendations)} recommendations")
        else:
            print(f"  Provider not available")


if __name__ == "__main__":
    asyncio.run(test_free_providers())
