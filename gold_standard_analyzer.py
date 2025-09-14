import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tri_model_api_config import get_system_config, get_trimodel_api_keys


class GoldStandardAIAnalyzer:
    """
    Advanced AI analyzer for MLB betting opportunities using multiple AI models.
    Implements Kelly Criterion and professional risk management.
    """

    def __init__(self, api_keys: dict[str, str]):
        self.api_keys = api_keys
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.config = get_system_config()

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            connector=aiohttp.TCPConnector(limit=100),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def _is_rate_limit(self, exception):
        """Check if exception is a rate limit error."""
        return (
            isinstance(exception, aiohttp.ClientResponseError)
            and exception.status == 429
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(
            lambda e: isinstance(e, aiohttp.ClientResponseError) and e.status == 429
        ),
        reraise=True,
    )
    async def _call_claude_api(self, prompt: str) -> dict[str, Any]:
        """Call Anthropic Claude API with rate limiting and retry logic."""
        if not self.api_keys.get("claude_api"):
            return {"error": "Claude API key not configured"}

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_keys["claude_api"],
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            async with self.session.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=payload
            ) as response:
                if response.status == 429:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=429,
                        message="Rate limit exceeded",
                    )
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            self.logger.error(f"Claude API error: {e}")
            return {"error": str(e)}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(
            lambda e: isinstance(e, aiohttp.ClientResponseError) and e.status == 429
        ),
        reraise=True,
    )
    async def _call_openai_api(self, prompt: str) -> dict[str, Any]:
        """Call OpenAI API with rate limiting and retry logic."""
        if not self.api_keys.get("openai_api"):
            return {"error": "OpenAI API key not configured"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['openai_api']}",
        }

        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.1,
        }

        try:
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status == 429:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=429,
                        message="Rate limit exceeded",
                    )
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return {"error": str(e)}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(
            lambda e: isinstance(e, aiohttp.ClientResponseError) and e.status == 429
        ),
        reraise=True,
    )
    async def _call_grok_api(self, prompt: str) -> dict[str, Any]:
        """Call Grok API with rate limiting and retry logic."""
        if not self.api_keys.get("grok_api"):
            return {"error": "Grok API key not configured"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['grok_api']}",
        }

        # Try multiple model names in order of preference
        models_to_try = ["grok-beta", "grok-2-latest", "grok-2", "grok-3-mini-beta"]

        for model_name in models_to_try:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.1,
            }

            try:
                async with self.session.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status == 404 and model_name != models_to_try[-1]:
                        # Try next model if 404 and not the last model
                        self.logger.warning(
                            f"Model {model_name} not found, trying next..."
                        )
                        continue
                    elif response.status == 429:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=429,
                            message="Rate limit exceeded",
                        )

                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Grok API success with model: {model_name}")
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(
                            f"Grok API error {response.status} with {model_name}: {error_text}"
                        )

            except Exception as e:
                self.logger.error(f"Grok API error with {model_name}: {e}")
                if model_name == models_to_try[-1]:
                    # Return error if it's the last model
                    return {"error": f"All Grok models failed. Last error: {str(e)}"}
                continue

        return {"error": "All Grok models failed with 404 errors"}

    async def analyze_single_opportunity(
        self, opportunity: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Analyze a single betting opportunity using all AI models."""
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(opportunity)

            # Call all AI models concurrently using asyncio.gather
            results = {}

            # Create list of coroutines to run
            tasks = []
            model_names = []

            if self.api_keys.get("claude_api"):
                tasks.append(self._call_claude_api(prompt))
                model_names.append("claude")

            if self.api_keys.get("openai_api"):
                tasks.append(self._call_openai_api(prompt))
                model_names.append("openai")

            # Only add Grok if we have the API key
            if self.api_keys.get("grok_api"):
                tasks.append(self._call_grok_api(prompt))
                model_names.append("grok")

            if not tasks:
                self.logger.warning("No API keys configured")
                return None

            # Execute all API calls concurrently
            try:
                api_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results with better error handling
                successful_results = 0
                for i, result in enumerate(api_results):
                    model_name = model_names[i]
                    if isinstance(result, Exception):
                        self.logger.error(f"Error calling {model_name} API: {result}")
                        results[model_name] = {"error": str(result)}
                    elif "error" in result:
                        self.logger.warning(
                            f"{model_name} API returned error: {result['error']}"
                        )
                        results[model_name] = result
                    else:
                        results[model_name] = result
                        successful_results += 1

                # Log success rate
                if successful_results > 0:
                    self.logger.info(
                        f"Successfully called {successful_results}/{len(tasks)} AI APIs"
                    )
                else:
                    self.logger.warning("All AI API calls failed")

            except Exception as e:
                self.logger.error(f"Error in concurrent API calls: {e}")
                return None

            # Process and aggregate results
            analysis = self._aggregate_ai_responses(results, opportunity)

            # Apply Kelly Criterion and risk management
            if analysis:
                return self._apply_kelly_criterion(analysis, opportunity)

            return None

        except Exception as e:
            self.logger.error(
                f"Error analyzing opportunity {opportunity.get('game_id', 'unknown')}: {e}"
            )
            return None

    def _create_analysis_prompt(self, opportunity: dict[str, Any]) -> str:
        """Create analysis prompt for AI models."""
        game_info = f"""
Analyze this MLB betting opportunity:

Game: {opportunity.get('away_team', 'Unknown')} @ {opportunity.get('home_team', 'Unknown')}
Bet Type: {opportunity.get('bet_type', 'Unknown')}
Selection: {opportunity.get('selection', 'Unknown')}
Odds: {opportunity.get('odds', 'Unknown')}
Bookmaker: {opportunity.get('bookmaker', 'Unknown')}

Please provide:
1. Win probability assessment (0-1)
2. Confidence level (0-1)
3. Key factors supporting/opposing this bet
4. Risk assessment
5. Overall recommendation (BET/PASS)

Respond in JSON format with these exact keys:
{{
    "win_probability": 0.XX,
    "confidence": 0.XX,
    "key_factors": ["factor1", "factor2"],
    "risk_level": "LOW/MEDIUM/HIGH",
    "recommendation": "BET/PASS",
    "reasoning": "brief explanation"
}}
"""
        return game_info

    def _aggregate_ai_responses(
        self, results: dict[str, dict], opportunity: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Aggregate responses from multiple AI models."""
        valid_responses = []

        for model_name, result in results.items():
            if "error" in result:
                continue

            try:
                # Extract content based on model response format
                content = ""
                if model_name == "claude" and "content" in result:
                    # Handle both list and string formats for Claude responses
                    if (
                        isinstance(result["content"], list)
                        and len(result["content"]) > 0
                    ):
                        content = (
                            result["content"][0].get("text", "")
                            if isinstance(result["content"][0], dict)
                            else str(result["content"][0])
                        )
                    elif isinstance(result["content"], str):
                        content = result["content"]
                elif model_name in ["openai", "grok"] and "choices" in result:
                    content = result["choices"][0]["message"]["content"]

                if content:
                    # Try to parse JSON from content
                    analysis = self._parse_ai_response(content)
                    if analysis:
                        analysis["source"] = model_name
                        valid_responses.append(analysis)

            except Exception as e:
                self.logger.error(f"Error parsing {model_name} response: {e}")
                continue

        if not valid_responses:
            return None

        # Aggregate the responses
        return self._merge_ai_analyses(valid_responses, opportunity)

    def _parse_ai_response(self, content: str) -> dict[str, Any] | None:
        """Parse AI response content to extract analysis."""
        try:
            # Try to find JSON in the response
            import re

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        # Fallback: extract key information using regex
        try:
            analysis = {}

            # Extract win probability
            prob_match = re.search(r'"win_probability":\s*([\d.]+)', content)
            if prob_match:
                analysis["win_probability"] = float(prob_match.group(1))

            # Extract confidence
            conf_match = re.search(r'"confidence":\s*([\d.]+)', content)
            if conf_match:
                analysis["confidence"] = float(conf_match.group(1))

            # Extract recommendation
            rec_match = re.search(r'"recommendation":\s*"(BET|PASS)"', content)
            if rec_match:
                analysis["recommendation"] = rec_match.group(1)

            # Set defaults if not found
            analysis.setdefault("win_probability", 0.5)
            analysis.setdefault("confidence", 0.5)
            analysis.setdefault("recommendation", "PASS")
            analysis.setdefault("risk_level", "MEDIUM")
            analysis.setdefault("reasoning", "Automated analysis")

            return analysis if analysis else None

        except Exception as e:
            self.logger.error(f"Error parsing AI response: {e}")
            return None

    def _merge_ai_analyses(
        self, analyses: list[dict[str, Any]], opportunity: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge multiple AI analyses into a single recommendation."""
        if not analyses:
            return None

        # Calculate averages
        avg_win_prob = sum(a.get("win_probability", 0.5) for a in analyses) / len(
            analyses
        )
        avg_confidence = sum(a.get("confidence", 0.5) for a in analyses) / len(analyses)

        # Count recommendations
        bet_count = sum(1 for a in analyses if a.get("recommendation") == "BET")
        consensus_strength = bet_count / len(analyses)

        # Determine final recommendation
        final_recommendation = (
            "BET"
            if consensus_strength >= 0.6
            and avg_confidence >= self.config["confidence_threshold"]
            else "PASS"
        )

        return {
            "win_probability": avg_win_prob,
            "confidence": avg_confidence,
            "consensus_strength": consensus_strength,
            "recommendation": final_recommendation,
            "model_count": len(analyses),
            "bet_votes": bet_count,
            "opportunity": opportunity,
        }

    def _apply_kelly_criterion(
        self, analysis: dict[str, Any], opportunity: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Apply Kelly Criterion for position sizing."""
        try:
            win_prob = analysis["win_probability"]
            odds = float(opportunity.get("odds", 2.0))
            confidence = analysis["confidence"]

            # Kelly Criterion: f = (bp - q) / b
            # where b = odds - 1, p = win_probability, q = 1 - p
            b = odds - 1
            p = win_prob
            q = 1 - p

            if b <= 0:
                return None

            kelly_fraction = (b * p - q) / b

            # Apply conservative scaling (25% of Kelly)
            conservative_fraction = kelly_fraction * 0.25

            # Risk management constraints
            max_fraction = (
                self.config["max_units"]
                * self.config["base_unit_size"]
                / self.config["bankroll"]
            )
            final_fraction = min(
                conservative_fraction, max_fraction, 0.05
            )  # Max 5% of bankroll

            # Only recommend if edge is positive and meets thresholds
            edge = (odds * win_prob) - 1

            if (
                final_fraction > 0
                and edge > 0.03  # Minimum 3% edge
                and confidence >= self.config["confidence_threshold"]
                and analysis["recommendation"] == "BET"
            ):

                bet_size = final_fraction * self.config["bankroll"]
                units = bet_size / self.config["base_unit_size"]

                return {
                    **analysis,
                    "kelly_fraction": kelly_fraction,
                    "conservative_fraction": conservative_fraction,
                    "final_fraction": final_fraction,
                    "edge": edge,
                    "bet_size": bet_size,
                    "units": round(units, 2),
                    "expected_value": bet_size * edge,
                    "risk_level": self._assess_risk_level(final_fraction, confidence),
                    "opportunity": opportunity,
                }

        except Exception as e:
            self.logger.error(f"Error applying Kelly Criterion: {e}")

        return None

    def _assess_risk_level(self, fraction: float, confidence: float) -> str:
        """Assess risk level based on position size and confidence."""
        if fraction <= 0.01 and confidence >= 0.8:
            return "LOW"
        elif fraction <= 0.03 and confidence >= 0.7:
            return "MEDIUM"
        else:
            return "HIGH"

    async def analyze_opportunities_concurrently(
        self, opportunities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analyze multiple opportunities concurrently."""
        semaphore = asyncio.Semaphore(self.config["max_concurrent_requests"])

        async def analyze_with_semaphore(opportunity):
            async with semaphore:
                return await self.analyze_single_opportunity(opportunity)

        tasks = [analyze_with_semaphore(opp) for opp in opportunities]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        valid_results = []
        for result in results:
            if isinstance(result, dict) and result is not None:
                valid_results.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Analysis exception: {result}")

        return valid_results
