#!/usr/bin/env python3
"""
Enhanced NFL Narrative Scraper
================================
Scrapes public sentiment from Reddit, YouTube, Twitter to understand game narratives.
Adds conspiracy scoring and betting recommendations.
"""

import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class SimpleNarrativeScraper:
    """Enhanced scraper for NFL game narratives with multi-source intelligence."""
    
    RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}

    def __init__(self, cache_ttl: int = 1800, request_timeout: float = 8.0, max_retries: int = 2):
        self.reddit_base = "https://www.reddit.com"
        self.youtube_key = os.getenv("YOUTUBE_API_KEY")
        self.cache = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache_ttl = max(0, cache_ttl)
        self.request_timeout = max(1.0, request_timeout)
        self.max_retries = max(0, max_retries)
        self._managed_session = False
    
    async def __aenter__(self) -> "SimpleNarrativeScraper":
        await self._ensure_session()
        self._managed_session = True
        return self
    
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
        self._managed_session = False
    
    async def get_game_narrative(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """
        Get narrative for a specific game from multi-source intelligence.
        
        Returns narrative signals:
        - narrative_strength: 0-1 (how much media attention)
        - public_lean: 0-1 (0=heavy under, 1=heavy over, 0.5=neutral)
        - vegas_bait: bool (strong public lean suggesting trap)
        - storylines: list of detected storylines
        - conspiracy_score: 0-1 (overall manipulation probability)
        - sharp_vs_public: sentiment divergence
        - betting_recommendation: actionable pick
        """
        
        cache_key = f"{away_team}_{home_team}"
        cached_entry = self.cache.get(cache_key)
        if cached_entry:
            if isinstance(cached_entry, dict) and "data" in cached_entry and "timestamp" in cached_entry:
                age_seconds = (datetime.now() - cached_entry["timestamp"]).total_seconds()
                if self.cache_ttl and age_seconds <= self.cache_ttl:
                    logger.debug(
                        "Using cached narrative for %s @ %s (age=%.1fs)",
                        away_team,
                        home_team,
                        age_seconds,
                    )
                    return cached_entry["data"]
                logger.debug(
                    "Cache expired for %s @ %s (age=%.1fs, ttl=%ss)",
                    away_team,
                    home_team,
                    age_seconds,
                    self.cache_ttl,
                )
                self.cache.pop(cache_key, None)
            else:
                return cached_entry
        
        try:
            # Scrape all sources
            reddit_narrative = await self._scrape_reddit_nfl(home_team, away_team)
            youtube_data = await self._scrape_youtube_sentiment(home_team, away_team)
            twitter_data = await self._scrape_twitter_sentiment(home_team, away_team)
            
            # Combine all sources
            narrative = self._combine_narratives(reddit_narrative, youtube_data, twitter_data)
            
            if self.cache_ttl:
                self.cache[cache_key] = {
                    "timestamp": datetime.now(),
                    "data": narrative,
                }
            return narrative
        except Exception as e:
            logger.warning(f"Narrative scrape failed for {away_team}@{home_team}: {e}")
            return self._fallback_narrative(home_team, away_team)
    
    async def close(self) -> None:
        """Close the shared HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        expect: str = "json",
    ):
        session = await self._ensure_session()
        last_exc: Optional[Exception] = None
        backoff = 0.75
        
        for attempt in range(self.max_retries + 1):
            try:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_payload,
                ) as resp:
                    try:
                        resp.raise_for_status()
                    except aiohttp.ClientResponseError as exc:
                        if exc.status in self.RETRY_STATUSES and attempt < self.max_retries:
                            last_exc = exc
                            await asyncio.sleep(backoff * (2 ** attempt))
                            continue
                        raise
                    
                    if expect == "json":
                        return await resp.json()
                    if expect == "text":
                        return await resp.text()
                    return await resp.read()
            except (aiohttp.ClientResponseError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                retryable = isinstance(exc, aiohttp.ClientResponseError) and getattr(exc, "status", None) in self.RETRY_STATUSES
                if attempt < self.max_retries and (retryable or isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))):
                    await asyncio.sleep(backoff * (2 ** attempt))
                    continue
                raise
        
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Request failed for {url}")
    
    async def _scrape_reddit_nfl(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Scrape r/NFL for game discussion."""
        
        # Search Reddit for game thread
        search_query = f"{away_team} {home_team} game thread"
        url = f"{self.reddit_base}/r/nfl/search.json?q={search_query}&sort=new&limit=10"
        
        try:
            data = await self._request_with_retry(
                "GET",
                url,
                headers={"User-Agent": "NFLBettingBot/1.0"},
                expect="json",
            )
            posts = data.get("data", {}).get("children", []) if data else []
            
            if not posts:
                return self._fallback_narrative(home_team, away_team)
            
            # Analyze post titles and comments
            narrative = self._analyze_reddit_posts(posts, home_team, away_team)
            return narrative
        except Exception as e:
            logger.error(f"Reddit scrape error: {e}")
            return self._fallback_narrative(home_team, away_team)
    
    def _analyze_reddit_posts(self, posts: list, home_team: str, away_team: str) -> Dict[str, Any]:
        """Analyze Reddit posts for narrative signals."""
        
        # Expanded keyword detection
        over_keywords = [
            "shootout", "score", "offense", "explosive", "points", "high-scoring",
            "fireworks", "barn burner", "air raid", "throwing", "passing game",
            "fantasy points", "dfs", "stack", "pace", "tempo", "fast",
        ]
        
        under_keywords = [
            "defense", "grind", "low-scoring", "ugly", "defensive", "boring",
            "sloppy", "conservative", "run heavy", "clock control", "field position",
            "weather", "wind", "rain", "snow", "cold", "mud", "trench battle",
        ]
        
        hype_keywords = [
            "primetime", "revenge", "playoff", "division", "rivalry", "must-win",
            "statement game", "prove it", "bounce back", "redemption",
        ]
        
        # NEW: Advanced storyline detection
        storyline_patterns = {
            "revenge_game": ["revenge", "got embarrassed", "blowout loss", "payback", "last time"],
            "division_rival": ["division", "rival", "hate each other", "division game", "afc/nfc"],
            "playoff_implications": ["playoff", "wildcard", "division lead", "must win", "elimination"],
            "coaching_drama": ["hot seat", "fire", "coach", "firing", "seat warm", "job on line"],
            "qb_narrative": ["qb duel", "matchup", "mvp", "mahomes", "allen", "brady", "legacy"],
            "weather_game": ["weather", "wind", "rain", "snow", "cold", "elements", "lambeau"],
            "trap_game": ["trap", "public", "square", "sharp", "vegas", "line movement", "everyone on"],
            "look_ahead": ["looking ahead", "trap game", "distracted", "next week", "sandwich"],
            "short_rest": ["thursday", "short week", "no rest", "tired", "banged up"],
            "prime_time": ["monday night", "thursday night", "sunday night", "snf", "mnf", "tnf"],
        }
        
        over_count = 0
        under_count = 0
        hype_count = 0
        total_words = 0
        
        storylines = []
        detected_storylines = set()
        
        for post in posts:
            post_data = post.get("data", {})
            title = post_data.get("title", "").lower()
            selftext = post_data.get("selftext", "").lower()
            text = f"{title} {selftext}"
            
            total_words += len(text.split())
            
            # Count sentiment keywords
            for keyword in over_keywords:
                if keyword in text:
                    over_count += 1
            
            for keyword in under_keywords:
                if keyword in text:
                    under_count += 1
            
            for keyword in hype_keywords:
                if keyword in text:
                    hype_count += 1
            
            # Detect storylines
            for storyline_name, patterns in storyline_patterns.items():
                for pattern in patterns:
                    if pattern in text:
                        detected_storylines.add(storyline_name)
                        break
        
        # Format storylines for output
        storyline_labels = {
            "revenge_game": "Revenge Game",
            "division_rival": "Division Rivalry",
            "playoff_implications": "Playoff Implications",
            "coaching_drama": "Coaching Hot Seat",
            "qb_narrative": "QB Duel",
            "weather_game": "Weather Factor",
            "trap_game": "Vegas Trap",
            "look_ahead": "Look-Ahead Spot",
            "short_rest": "Short Rest",
            "prime_time": "Prime Time",
        }
        
        for storyline in detected_storylines:
            storylines.append(storyline_labels.get(storyline, storyline))
        
        # Calculate narrative strength (media attention)
        narrative_strength = min(1.0, total_words / 500)  # More words = more attention
        narrative_strength = max(narrative_strength, 0.3)  # Minimum baseline
        
        # Calculate public lean (over vs under sentiment)
        total_sentiment = over_count + under_count
        if total_sentiment > 0:
            public_lean = over_count / total_sentiment
        else:
            public_lean = 0.5  # Neutral
        
        # Detect Vegas bait (strong public lean)
        vegas_bait = (public_lean > 0.7 or public_lean < 0.3) and total_sentiment > 3
        
        # Media hype
        media_hype = min(1.0, hype_count / 3)
        
        return {
            "narrative_strength": narrative_strength,
            "public_lean": public_lean,
            "vegas_bait": vegas_bait,
            "media_hype": media_hype,
            "storylines": storylines[:3],  # Top 3 storylines
            "sentiment_counts": {
                "over": over_count,
                "under": under_count,
                "hype": hype_count,
            }
        }
    
    async def _scrape_youtube_sentiment(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Scrape YouTube for game narratives and predictions."""
        
        if not self.youtube_key:
            logger.debug("No YouTube API key - skipping YouTube scraping")
            return {"videos_analyzed": 0, "narratives": {}, "narrative_strength": 0}
        
        search_query = f"{away_team} {home_team} NFL 2025 predictions picks"
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": search_query,
            "type": "video",
            "maxResults": 10,
            "order": "relevance",
            "key": self.youtube_key,
        }
        
        try:
            data = await self._request_with_retry(
                "GET",
                url,
                params=params,
                expect="json",
            )
            videos = data.get("items", []) if data else []
            return self._analyze_video_sentiment(videos)
        except Exception as e:
            logger.warning(f"YouTube scrape failed: {e}")
            return {"videos_analyzed": 0, "narratives": {}, "narrative_strength": 0}
    
    def _analyze_video_sentiment(self, videos: List[Dict]) -> Dict[str, Any]:
        """Analyze YouTube video titles/descriptions for narratives."""
        
        narratives = {
            "revenge_game": 0,
            "trap_game": 0,
            "media_hype": 0,
            "sharp_lean": 0,
            "public_favorite": 0,
        }
        
        keywords = {
            "revenge_game": ["revenge", "payback", "rematch"],
            "trap_game": ["trap", "letdown", "sleeper", "upset alert"],
            "media_hype": ["best", "unstoppable", "elite", "dominant"],
            "sharp_lean": ["sharp", "value", "line movement", "wiseguy"],
            "public_favorite": ["everyone", "public", "lock", "easy win"],
        }
        
        for video in videos:
            title = video.get("snippet", {}).get("title", "").lower()
            description = video.get("snippet", {}).get("description", "").lower()
            text = f"{title} {description}"
            
            for narrative, terms in keywords.items():
                if any(term in text for term in terms):
                    narratives[narrative] += 1
        
        total = len(videos) if videos else 1
        sentiment_scores = {k: v / total for k, v in narratives.items()}
        
        return {
            "videos_analyzed": len(videos),
            "narratives": sentiment_scores,
            "narrative_strength": max(sentiment_scores.values()) if sentiment_scores else 0,
        }
    
    async def _scrape_twitter_sentiment(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Scrape Twitter/X via Nitter for sharp vs public sentiment."""
        
        query = f"{away_team} {home_team} betting".replace(" ", "%20")
        nitter_instances = [
            "https://nitter.net",
            "https://nitter.privacydev.net",
            "https://nitter.poast.org",
        ]
        
        for nitter_url in nitter_instances:
            try:
                url = f"{nitter_url}/search?f=tweets&q={query}"
                html = await self._request_with_retry(
                    "GET",
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    expect="text",
                )
                
                # Detect sharp vs public keywords
                sharp_keywords = ["sharp", "wiseguy", "vegas", "line move", "smart money"]
                public_keywords = ["lock", "easy money", "smash", "hammer", "bet house", "mortgage"]
                
                html_lower = html.lower()
                sharp_count = sum(html_lower.count(k) for k in sharp_keywords)
                public_count = sum(html_lower.count(k) for k in public_keywords)
                total_mentions = sharp_count + public_count
                
                tweet_count = html.count('class="tweet-content"')
                
                logger.debug(f"Twitter via {nitter_url}: {tweet_count} tweets, sharp={sharp_count}, public={public_count}")

                if total_mentions == 0:
                    return {
                        "tweet_count": tweet_count,
                        "sharp_mentions": sharp_count,
                        "public_mentions": public_count,
                        "sharp_vs_public": 0.5,
                    }
                
                return {
                    "tweet_count": tweet_count,
                    "sharp_mentions": sharp_count,
                    "public_mentions": public_count,
                    "sharp_vs_public": sharp_count / total_mentions,
                }
            
            except Exception as e:
                logger.debug(f"Nitter instance {nitter_url} failed: {e}")
                continue
        
        logger.debug("All Nitter instances failed - skipping Twitter")
        return {"tweet_count": 0, "sharp_mentions": 0, "public_mentions": 0, "sharp_vs_public": 0.5}
    
    def _combine_narratives(
        self,
        reddit_data: Dict[str, Any],
        youtube_data: Dict[str, Any],
        twitter_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine narratives from all sources and calculate conspiracy score."""
        
        # Base data from Reddit
        combined = reddit_data.copy()
        
        reddit_counts = reddit_data.get("sentiment_counts", {}) or {}
        reddit_mentions = float(sum(reddit_counts.values()))
        youtube_videos = float(youtube_data.get("videos_analyzed", 0) or 0)
        twitter_activity = float(twitter_data.get("tweet_count", 0) or 0)
        
        reddit_weight = min(1.0, reddit_mentions / 8.0) if reddit_mentions else 0.0
        youtube_weight = min(1.0, youtube_videos / 6.0) if youtube_videos else 0.0
        twitter_weight = min(1.0, twitter_activity / 25.0) if twitter_activity else 0.0
        
        total_weight = reddit_weight + youtube_weight + twitter_weight
        overall_weight = min(1.0, total_weight / 3.0) if total_weight else 0.0
        
        confidence_weights = {
            "reddit": reddit_weight,
            "youtube": youtube_weight,
            "twitter": twitter_weight,
            "overall": overall_weight,
        }
        combined["confidence_weights"] = confidence_weights
        
        # Add YouTube narratives
        youtube_narratives = youtube_data.get("narratives", {})
        combined["youtube_signals"] = youtube_narratives
        
        # Add Twitter sharp vs public
        combined["sharp_vs_public"] = twitter_data.get("sharp_vs_public", 0.5)
        combined["twitter_activity"] = twitter_data.get("tweet_count", 0)
        combined["sharp_vs_public_confidence"] = twitter_weight
        
        # Enhanced narrative strength (combine Reddit + YouTube)
        reddit_strength = reddit_data.get("narrative_strength", 0)
        youtube_strength = youtube_data.get("narrative_strength", 0)
        reddit_strength_weight = max(0.1, reddit_weight) if reddit_strength else reddit_weight
        youtube_strength_weight = max(0.1, youtube_weight) if youtube_strength else youtube_weight
        strength_denominator = reddit_strength_weight + youtube_strength_weight
        if strength_denominator:
            combined_strength = (
                reddit_strength * reddit_strength_weight
                + youtube_strength * youtube_strength_weight
            ) / strength_denominator
        else:
            combined_strength = max(reddit_strength, youtube_strength * 0.8)
        combined["narrative_strength"] = min(1.0, max(0.0, combined_strength))
        
        # Down-weight public lean confidence when Reddit data is sparse
        reddit_public_lean = reddit_data.get("public_lean", 0.5)
        adjusted_public_lean = 0.5 + reddit_weight * (reddit_public_lean - 0.5)
        combined["public_lean"] = min(1.0, max(0.0, adjusted_public_lean))
        combined["public_lean_confidence"] = reddit_weight
        
        # Calculate conspiracy score
        conspiracy_score = self._calculate_conspiracy_score(
            reddit_data,
            youtube_data,
            twitter_data,
            confidence_weights
        )
        combined["conspiracy_score"] = conspiracy_score
        
        # Generate betting recommendation
        recommendation = self._generate_betting_rec(combined)
        combined["betting_recommendation"] = recommendation
        
        return combined
    
    def _calculate_conspiracy_score(
        self,
        reddit_data: Dict,
        youtube_data: Dict,
        twitter_data: Dict,
        weights: Dict[str, float],
    ) -> float:
        """Calculate overall conspiracy probability from all sources."""
        
        score = 0.0
        reddit_weight = weights.get("reddit", 0.0)
        youtube_weight = weights.get("youtube", 0.0)
        twitter_weight = weights.get("twitter", 0.0)
        
        # Reddit vegas bait signal
        if reddit_data.get("vegas_bait", False):
            score += 0.3 * reddit_weight
        
        # High media hype
        if reddit_data.get("media_hype", 0) > 0.6:
            score += 0.2 * reddit_weight
        
        # YouTube trap game signal
        youtube_narratives = youtube_data.get("narratives", {})
        if youtube_narratives.get("trap_game", 0) > 0.25:
            score += 0.25 * youtube_weight
        
        # Public favorite with high hype
        if youtube_narratives.get("public_favorite", 0) > 0.35:
            score += 0.15 * youtube_weight
        
        # Sharp vs public divergence
        sharp_vs_public = twitter_data.get("sharp_vs_public", 0.5)
        if sharp_vs_public < 0.3 or sharp_vs_public > 0.7:
            score += 0.2 * twitter_weight
        
        return min(1.0, score)
    
    def _generate_betting_rec(self, narrative_data: Dict) -> str:
        """Generate actionable betting recommendation."""
        
        public_lean = narrative_data.get("public_lean", 0.5)
        vegas_bait = narrative_data.get("vegas_bait", False)
        conspiracy_score = narrative_data.get("conspiracy_score", 0)
        
        # Strong conspiracy + public over lean = fade to under
        if conspiracy_score > 0.6 and public_lean > 0.7:
            return "FADE_PUBLIC_UNDER"
        
        # Strong conspiracy + public under lean = fade to over
        elif conspiracy_score > 0.6 and public_lean < 0.3:
            return "FADE_PUBLIC_OVER"
        
        # Vegas bait detected
        elif vegas_bait:
            if public_lean > 0.65:
                return "TRAP_LEAN_UNDER"
            else:
                return "TRAP_LEAN_OVER"
        
        # Sharp money divergence
        sharp_vs_public = narrative_data.get("sharp_vs_public", 0.5)
        if sharp_vs_public > 0.7:
            return "FOLLOW_SHARP"
        
        return "NO_CLEAR_EDGE"
    
    def _fallback_narrative(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Fallback narrative based on team profiles."""
        
        # High-profile teams (big markets, dynasties)
        big_market = {"KC", "LAC", "DAL", "NE", "SF", "PHI", "NYG", "NYJ", "CHI", "MIA", "GB", "PIT"}
        high_offense = {"KC", "BUF", "MIA", "LAC", "CIN", "DAL", "SF", "PHI", "DET"}
        
        is_big_game = home_team in big_market or away_team in big_market
        has_offense = home_team in high_offense or away_team in high_offense
        
        narrative_strength = 0.7 if is_big_game else 0.4
        public_lean = 0.6 if has_offense else 0.5  # Slight over bias for offensive teams
        vegas_bait = is_big_game and has_offense  # Public loves high-profile offensive games
        
        storylines = []
        if is_big_game:
            storylines.append("Primetime")
        if has_offense:
            storylines.append("High-Scoring")
        
        confidence_weights = {
            "reddit": 0.0,
            "youtube": 0.0,
            "twitter": 0.0,
            "overall": 0.0,
        }
        
        return {
            "narrative_strength": narrative_strength,
            "public_lean": public_lean,
            "vegas_bait": vegas_bait,
            "media_hype": 0.6 if is_big_game else 0.3,
            "storylines": storylines,
            "sentiment_counts": {"over": 0, "under": 0, "hype": 0},
            "conspiracy_score": 0.3 if vegas_bait else 0.1,
            "sharp_vs_public": 0.5,
            "twitter_activity": 0,
            "youtube_signals": {},
            "betting_recommendation": "NO_CLEAR_EDGE",
            "confidence_weights": confidence_weights,
            "public_lean_confidence": 0.0,
            "sharp_vs_public_confidence": 0.0,
        }


async def test_scraper():
    """Test the narrative scraper."""
    scraper = SimpleNarrativeScraper()
    
    # Test on LAC vs MIN
    narrative = await scraper.get_game_narrative("LAC", "MIN")
    
    print("="*60)
    print("MIN @ LAC NARRATIVE ANALYSIS")
    print("="*60)
    print(f"Narrative Strength: {narrative['narrative_strength']:.2f}")
    print(f"Public Lean: {narrative['public_lean']:.2f} (0=under, 1=over)")
    print(f"Vegas Bait: {narrative['vegas_bait']}")
    print(f"Media Hype: {narrative['media_hype']:.2f}")
    print(f"Storylines: {', '.join(narrative['storylines'])}")
    print(f"Sentiment: Over={narrative['sentiment_counts']['over']}, Under={narrative['sentiment_counts']['under']}")


if __name__ == "__main__":
    asyncio.run(test_scraper())
