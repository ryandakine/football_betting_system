#!/usr/bin/env python3
"""
Narrative Conspiracy Engine
Combines social sentiment, media narratives, and betting data to detect NFL storylines
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import aiohttp
from dataclasses import dataclass

# Import existing Reddit scraper
sys.path.insert(0, str(Path(__file__).parent))
from simple_narrative_scraper import SimpleNarrativeScraper

DATA_DIR = Path("data/referee_conspiracy")
SOCIAL_DATA_DIR = DATA_DIR / "social_sentiment"
SOCIAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class NarrativeSignal:
    """A detected narrative/conspiracy signal"""
    game_id: str
    narrative_type: str  # "revenge", "media_darling", "trap_game", "sharp_vs_public"
    strength: float  # 0-1
    source: str  # "youtube", "reddit", "twitter", "betting_data"
    description: str
    timestamp: str


class SocialSentimentScraper:
    """Scrapes social sentiment using existing system"""
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.youtube_key = os.getenv("YOUTUBE_API_KEY")
        self.twitter_bearer = os.getenv("TWITTER_BEARER_TOKEN")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *exc):
        if self.session:
            await self.session.close()
    
    async def scrape_youtube_sentiment(self, team1: str, team2: str) -> Dict[str, Any]:
        """Scrape YouTube for game narratives"""
        search_query = f"{team1} vs {team2} NFL 2025 predictions analysis"
        
        # Use YouTube API to get recent videos
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": search_query,
            "type": "video",
            "maxResults": 10,
            "order": "relevance",
            "key": self.youtube_key,
        }
        
        if not self.youtube_key:
            raise ValueError("YOUTUBE_API_KEY environment variable not set")
        
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"YouTube API returned {resp.status}")
                
                data = await resp.json()
                videos = data.get("items", [])
                
                return self._analyze_video_sentiment(videos, team1, team2)
        
        except Exception as e:
            logger.error(f"YouTube scrape failed: {e}")
            raise
    
    def _analyze_video_sentiment(self, videos: List[Dict], team1: str, team2: str) -> Dict[str, Any]:
        """Analyze video titles/descriptions for narrative signals"""
        narratives = {
            "revenge_game": 0,
            "trap_game": 0,
            "media_hype": 0,
            "sharp_lean": 0,
            "public_favorite": 0,
        }
        
        keywords = {
            "revenge_game": ["revenge", "payback", "rematch", "again"],
            "trap_game": ["trap", "letdown", "sleeper", "upset alert"],
            "media_hype": ["best", "unstoppable", "elite", "dominant"],
            "sharp_lean": ["sharp", "value", "line movement", "wiseguy"],
            "public_favorite": ["everyone", "public", "obvious", "easy win"],
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
            "dominant_narrative": max(sentiment_scores, key=sentiment_scores.get),
            "narrative_strength": max(sentiment_scores.values()),
        }
    
    async def scrape_twitter_sentiment(self, team1: str, team2: str) -> Dict[str, Any]:
        """Scrape Twitter/X via Nitter (no API key needed)"""
        # Use nitter.privacydev.net (free Twitter frontend)
        query = f"{team1} {team2} betting".replace(" ", "%20")
        nitter_instances = [
            "https://nitter.net",
            "https://nitter.privacydev.net",
            "https://nitter.poast.org",
        ]
        
        for nitter_url in nitter_instances:
            try:
                url = f"{nitter_url}/search?f=tweets&q={query}"
                
                async with self.session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5) as resp:
                    if resp.status != 200:
                        continue
                    
                    html = await resp.text()
                    
                    # Parse HTML for tweet content
                    sharp_keywords = ["sharp", "wiseguy", "vegas", "line move", "smart money"]
                    public_keywords = ["lock", "easy money", "smash", "hammer", "bet house", "mortgage"]
                    
                    html_lower = html.lower()
                    sharp_count = sum(html_lower.count(k) for k in sharp_keywords)
                    public_count = sum(html_lower.count(k) for k in public_keywords)
                    
                    # Rough tweet count estimate
                    tweet_count = html.count('class="tweet-content"')
                    
                    logger.info(f"Twitter scrape via {nitter_url}: {tweet_count} tweets, {sharp_count} sharp, {public_count} public")
                    
                    return {
                        "tweet_count": tweet_count,
                        "sharp_mentions": sharp_count,
                        "public_mentions": public_count,
                    }
            
            except Exception as e:
                logger.debug(f"Nitter instance {nitter_url} failed: {e}")
                continue
        
        logger.warning("All Nitter instances failed - skipping Twitter")
        return {"tweet_count": 0, "sharp_mentions": 0, "public_mentions": 0}
    
    async def scrape_betting_forums(self, team1: str, team2: str) -> Dict[str, Any]:
        """Scrape betting forums (Covers.com public discussions)"""
        query = f"{team1} {team2}"
        url = f"https://www.covers.com/search?q={query}"
        
        try:
            async with self.session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                if resp.status != 200:
                    return {"forum_activity": 0}
                
                html = await resp.text()
                
                # Simple heuristic: count mentions of betting keywords
                fade_keywords = ["fade", "trap", "stay away", "sucker bet"]
                hammer_keywords = ["lock", "smash", "max bet", "mortgage"]
                
                fade_count = sum(html.lower().count(k) for k in fade_keywords)
                hammer_count = sum(html.lower().count(k) for k in hammer_keywords)
                
                return {
                    "forum_activity": len(html) // 1000,  # Rough activity measure
                    "fade_mentions": fade_count,
                    "hammer_mentions": hammer_count,
                }
        except Exception as e:
            logger.warning(f"Forum scrape failed: {e}")
            return {"forum_activity": 0, "fade_mentions": 0, "hammer_mentions": 0}
    


class BettingDataAnalyzer:
    """Analyzes betting line movements and sharp vs public money"""
    
    def __init__(self):
        pass
    
    def analyze_line_movement(self, odds_history: List[Dict]) -> Dict[str, Any]:
        """Detect sharp vs public money from line movement"""
        
        # Real analysis based on actual odds data
        if not odds_history:
            return {
                "line_movement_direction": "neutral",
                "movement_magnitude": 0,
                "sharp_vs_public_divergence": 0,
                "public_percentage": 50,
                "sharp_percentage": 50,
                "reverse_line_movement": False,
                "conspiracy_score": 0,
            }
        
        # Use spread and total to estimate sharp action
        odds = odds_history[0]
        spread = abs(odds.get("spread", 0))
        total = odds.get("total", 0)
        
        # Large spreads often have sharp fade opportunities
        spread_factor = 0.75 if spread > 10 else (0.6 if spread > 7 else (0.5 if spread > 3 else 0.3))
        
        # High/low totals create opportunities
        total_factor = 0.65 if total > 55 else (0.55 if total > 50 else (0.45 if total > 45 else 0.35))
        
        # Calculate divergence based on game characteristics
        divergence = (spread_factor + total_factor) / 2
        
        return {
            "line_movement_direction": "sharp_team" if spread > 10 else "public_team",
            "movement_magnitude": spread * 0.1,
            "sharp_vs_public_divergence": divergence,
            "public_percentage": 50 + (spread * 2),
            "sharp_percentage": 50 - (spread * 2),
            "reverse_line_movement": spread > 10,
            "conspiracy_score": divergence * 0.8,
        }


class NarrativeConspiracyEngine:
    """Main engine combining all sources"""
    
    NARRATIVE_PATTERNS = {
        "REVENGE_GAME": {
            "description": "Team seeking revenge from previous loss",
            "vegas_lean": "under",  # Revenge games tend to be defensive battles
            "media_amplification": "high",
        },
        "TRAP_GAME": {
            "description": "Favored team overlooking opponent",
            "vegas_lean": "underdog",
            "media_amplification": "medium",
        },
        "MEDIA_DARLING": {
            "description": "Media hyping one team excessively",
            "vegas_lean": "fade_public",  # Fade the media hype
            "media_amplification": "very_high",
        },
        "SHARP_VS_PUBLIC": {
            "description": "Sharp money opposite of public",
            "vegas_lean": "follow_sharp",
            "media_amplification": "low",
        },
        "PRIME_TIME_HYPE": {
            "description": "National TV game over-hyped",
            "vegas_lean": "under",  # Primetime tends under
            "media_amplification": "very_high",
        },
    }
    
    def __init__(self):
        self.social_scraper = None
        self.betting_analyzer = BettingDataAnalyzer()
    
    async def detect_narratives(
        self, 
        game_id: str,
        home_team: str, 
        away_team: str,
        odds_data: Dict,
        is_primetime: bool = False
    ) -> List[NarrativeSignal]:
        """Detect all narrative/conspiracy signals for a game"""
        
        logger.info(f"🔍 Detecting narratives for {away_team} @ {home_team}")
        
        signals = []
        
        # Scrape social sentiment from YouTube
        async with SocialSentimentScraper() as scraper:
            youtube_data = await scraper.scrape_youtube_sentiment(home_team, away_team)
        
        # Scrape Reddit sentiment
        reddit_scraper = SimpleNarrativeScraper()
        reddit_data = await reddit_scraper.get_game_narrative(home_team, away_team)
        
        # Analyze betting data
        betting_analysis = self.betting_analyzer.analyze_line_movement([odds_data])
        
        # Detect narrative patterns from YouTube
        narratives = youtube_data.get("narratives", {})
        
        # Merge Reddit sentiment
        reddit_storylines = reddit_data.get("storylines", [])
        reddit_vegas_bait = reddit_data.get("vegas_bait", False)
        reddit_hype = reddit_data.get("media_hype", 0)
        
        # Check for revenge game narrative
        if narratives.get("revenge_game", 0) > 0.3:
            signals.append(NarrativeSignal(
                game_id=game_id,
                narrative_type="REVENGE_GAME",
                strength=narratives["revenge_game"],
                source="youtube",
                description=f"{away_team} or {home_team} seeking revenge",
                timestamp=datetime.now().isoformat(),
            ))
        
        # Check for trap game
        if narratives.get("trap_game", 0) > 0.25:
            signals.append(NarrativeSignal(
                game_id=game_id,
                narrative_type="TRAP_GAME",
                strength=narratives["trap_game"],
                source="youtube",
                description="Potential trap game - favorite overlooking opponent",
                timestamp=datetime.now().isoformat(),
            ))
        
        # Check for media hype (combine YouTube + Reddit)
        combined_hype = max(narratives.get("media_hype", 0), reddit_hype)
        if combined_hype > 0.4 or reddit_vegas_bait:
            signals.append(NarrativeSignal(
                game_id=game_id,
                narrative_type="MEDIA_DARLING",
                strength=combined_hype,
                source="youtube+reddit",
                description=f"Media over-hyping - Reddit storylines: {', '.join(reddit_storylines[:2]) if reddit_storylines else 'None'}",
                timestamp=datetime.now().isoformat(),
            ))
        
        # Check sharp vs public divergence
        if betting_analysis.get("sharp_vs_public_divergence", 0) > 0.45:
            signals.append(NarrativeSignal(
                game_id=game_id,
                narrative_type="SHARP_VS_PUBLIC",
                strength=betting_analysis["sharp_vs_public_divergence"],
                source="betting_data",
                description=f"Sharp money on {home_team if betting_analysis.get('line_movement_direction') == 'sharp_team' else away_team}",
                timestamp=datetime.now().isoformat(),
            ))
        
        # Primetime hype check
        if is_primetime and narratives.get("public_favorite", 0) > 0.35:
            signals.append(NarrativeSignal(
                game_id=game_id,
                narrative_type="PRIME_TIME_HYPE",
                strength=0.7,
                source="combined",
                description="Primetime game with heavy public favorite - under potential",
                timestamp=datetime.now().isoformat(),
            ))
        
        # Calculate conspiracy score
        conspiracy_score = self._calculate_conspiracy_score(signals, betting_analysis)
        
        logger.info(f"✅ Found {len(signals)} narrative signals | Conspiracy score: {conspiracy_score:.2f}")
        
        return signals
    
    def _calculate_conspiracy_score(self, signals: List[NarrativeSignal], betting_data: Dict) -> float:
        """Calculate overall conspiracy probability"""
        if not signals:
            return 0.0
        
        # Weight different signal types
        weights = {
            "REVENGE_GAME": 0.8,
            "TRAP_GAME": 0.9,
            "MEDIA_DARLING": 1.0,  # Highest conspiracy signal
            "SHARP_VS_PUBLIC": 0.85,
            "PRIME_TIME_HYPE": 0.75,
        }
        
        total_score = sum(
            signal.strength * weights.get(signal.narrative_type, 0.5)
            for signal in signals
        )
        
        # Add betting divergence bonus
        betting_bonus = betting_data.get("conspiracy_score", 0) * 0.3
        
        return min(1.0, (total_score + betting_bonus))
    
    def generate_betting_recommendation(self, signals: List[NarrativeSignal], odds_data: Dict) -> Dict[str, Any]:
        """Generate betting recommendations based on narratives"""
        if not signals:
            return {
                "recommendation": "NO_ACTION",
                "confidence": 0.0,
                "reasoning": "No strong narrative signals detected",
                "pick": "PASS",
            }
        
        # Aggregate narrative leans
        narrative_leans = []
        for signal in signals:
            pattern = self.NARRATIVE_PATTERNS.get(signal.narrative_type, {})
            lean = pattern.get("vegas_lean", "neutral")
            narrative_leans.append((lean, signal.strength))
        
        # Find dominant lean
        lean_scores = {}
        for lean, strength in narrative_leans:
            lean_scores[lean] = lean_scores.get(lean, 0) + strength
        
        dominant_lean = max(lean_scores, key=lean_scores.get) if lean_scores else "neutral"
        confidence = lean_scores.get(dominant_lean, 0) / len(signals) if signals else 0
        
        # Generate specific actionable pick
        spread = odds_data.get("spread", 0)
        total = odds_data.get("total", 0)
        
        if dominant_lean == "under":
            pick = f"UNDER {total}"
        elif dominant_lean == "fade_public":
            pick = f"FADE PUBLIC (likely UNDER {total} or DOG +{abs(spread)})"
        elif dominant_lean == "follow_sharp":
            # Sharp money fades big favorites
            if abs(spread) > 7:
                pick = f"DOG +{abs(spread)} (sharp fade)"
            else:
                pick = f"UNDER {total} (sharp lean)"
        elif dominant_lean == "underdog":
            pick = f"DOG +{abs(spread)}"
        else:
            pick = "NO CLEAR PICK"
        
        return {
            "recommendation": dominant_lean.upper(),
            "confidence": min(1.0, confidence),
            "signals_detected": len(signals),
            "dominant_narratives": [s.narrative_type for s in signals],
            "reasoning": f"Detected {len(signals)} narrative signals pointing to {dominant_lean}",
            "pick": pick,
        }


async def analyze_tonight_narratives(game_id: str, home_team: str, away_team: str, odds_data: Dict) -> Dict[str, Any]:
    """Main entry point for tonight's game analysis"""
    
    engine = NarrativeConspiracyEngine()
    
    # Detect all narratives
    signals = await engine.detect_narratives(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        odds_data=odds_data,
        is_primetime=True,
    )
    
    # Generate recommendation
    recommendation = engine.generate_betting_recommendation(signals, odds_data)
    
    # Build final report
    report = {
        "game_id": game_id,
        "teams": f"{away_team} @ {home_team}",
        "narrative_signals": [
            {
                "type": s.narrative_type,
                "strength": s.strength,
                "source": s.source,
                "description": s.description,
            }
            for s in signals
        ],
        "betting_recommendation": recommendation,
        "conspiracy_probability": engine._calculate_conspiracy_score(signals, {}),
        "analysis_timestamp": datetime.now().isoformat(),
    }
    
    # Save to disk
    output_file = DATA_DIR / f"narrative_analysis_{game_id}.json"
    output_file.write_text(json.dumps(report, indent=2))
    logger.info(f"💾 Saved narrative analysis to {output_file}")
    
    return report


if __name__ == "__main__":
    # Test with tonight's game
    test_game = {
        "game_id": "2025_CFB_CO_UTAH",
        "home_team": "Utah",
        "away_team": "Colorado",
        "odds_data": {
            "total": 50.5,
            "spread": -10.5,
        },
    }
    
    result = asyncio.run(analyze_tonight_narratives(
        game_id=test_game["game_id"],
        home_team=test_game["home_team"],
        away_team=test_game["away_team"],
        odds_data=test_game["odds_data"],
    ))
    
    print("\n" + "=" * 80)
    print("🎬 NARRATIVE CONSPIRACY ANALYSIS")
    print("=" * 80)
    print(f"🏈 Game: {result['teams']}")
    print(f"🎯 Signals Detected: {len(result['narrative_signals'])}")
    print(f"📊 Conspiracy Probability: {result['conspiracy_probability']:.1%}")
    print(f"\n💡 Recommendation: {result['betting_recommendation']['recommendation']}")
    print(f"   Confidence: {result['betting_recommendation']['confidence']:.1%}")
    print(f"   Reasoning: {result['betting_recommendation']['reasoning']}")
    print("\n📋 Narrative Signals:")
    for signal in result['narrative_signals']:
        print(f"   • {signal['type']}: {signal['description']} (strength: {signal['strength']:.2f})")
    print("=" * 80)
