#!/usr/bin/env python3
"""
Unified NFL Betting Intelligence System
========================================
Combines line movement tracking, weather analysis, injury reports, 
and narrative detection into a unified betting intelligence engine.

This is the "master brain" that synthesizes all signals.
"""

import asyncio
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import our intelligence modules
from line_movement_tracker import LineMovementTracker, LineMovement
from weather_integration import fetch_weather_for_games, WeatherConditions
from injury_tracker import fetch_injuries_for_games, TeamInjuryReport
from simple_narrative_scraper import SimpleNarrativeScraper

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/referee_conspiracy")


class UnifiedBettingIntelligence:
    """Unified intelligence system combining all signals"""
    
    def __init__(self):
        self.line_tracker = LineMovementTracker()
        self.narrative_scraper = SimpleNarrativeScraper()
        
        # Storage
        self.line_movements: Dict[str, List[LineMovement]] = {}
        self.weather_reports: Dict[str, WeatherConditions] = {}
        self.injury_reports: Dict[str, TeamInjuryReport] = {}
        self.narrative_data: Dict[str, dict] = {}
    
    async def analyze_game(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        total: float,
        spread: float,
    ) -> Dict[str, Any]:
        """
        Analyze a single game with all intelligence systems.
        
        Returns comprehensive betting recommendation.
        """
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 ANALYZING: {away_team} @ {home_team}")
        logger.info(f"   Total: {total} | Spread: {spread:+.1f}")
        logger.info(f"{'='*80}")
        
        # Gather all intelligence
        line_signals = self._get_line_signals(game_id)
        weather_signal = self._get_weather_signal(home_team)
        injury_signals = self._get_injury_signals(home_team, away_team)
        narrative_signal = await self._get_narrative_signal(home_team, away_team)
        
        # Synthesize signals
        recommendation = self._synthesize_signals(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            total=total,
            spread=spread,
            line_signals=line_signals,
            weather_signal=weather_signal,
            injury_signals=injury_signals,
            narrative_signal=narrative_signal,
        )
        
        return recommendation
    
    def _get_line_signals(self, game_id: str) -> Dict[str, Any]:
        """Get line movement signals"""
        
        movements = self.line_tracker.detect_movements(game_id)
        
        if not movements:
            return {
                "has_movement": False,
                "sharp_detected": False,
                "steam_detected": False,
                "edge": 0.0,
            }
        
        # Find highest confidence movement
        best_movement = max(movements, key=lambda m: m.sharp_confidence)
        
        has_sharp = best_movement.sharp_confidence >= 0.6
        has_steam = any(m.is_steam_move for m in movements)
        
        # Calculate edge from line movement
        edge = 0.0
        if has_sharp:
            edge = best_movement.sharp_confidence * 2.0  # Scale to 0-2 points
        
        logger.info(f"📊 LINE MOVEMENT: Sharp={has_sharp}, Steam={has_steam}, Edge={edge:+.1f}")
        
        return {
            "has_movement": True,
            "sharp_detected": has_sharp,
            "steam_detected": has_steam,
            "edge": edge,
            "best_movement": best_movement.to_dict() if movements else None,
            "movements": [m.to_dict() for m in movements],
        }
    
    def _get_weather_signal(self, home_team: str) -> Dict[str, Any]:
        """Get weather impact signal"""
        
        weather = self.weather_reports.get(home_team)
        
        if not weather or weather.is_dome:
            return {
                "has_impact": False,
                "adjustment": 0.0,
                "severity": 0.0,
            }
        
        logger.info(
            f"🌤️  WEATHER: {weather.recommendation}, "
            f"Adjustment={weather.total_adjustment:+.1f}, "
            f"Severity={weather.weather_severity:.0%}"
        )
        
        return {
            "has_impact": weather.weather_severity >= 0.3,
            "adjustment": weather.total_adjustment,
            "severity": weather.weather_severity,
            "recommendation": weather.recommendation,
            "conditions": weather.description,
        }
    
    def _get_injury_signals(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Get injury impact signals"""
        
        home_injuries = self.injury_reports.get(home_team)
        away_injuries = self.injury_reports.get(away_team)
        
        # Net impact
        home_impact = home_injuries.total_impact if home_injuries else 0.0
        away_impact = away_injuries.total_impact if away_injuries else 0.0
        
        net_total_impact = home_impact + away_impact
        
        home_spread = home_injuries.spread_impact if home_injuries else 0.0
        away_spread = away_injuries.spread_impact if away_injuries else 0.0
        
        # Positive spread impact hurts that team
        net_spread_impact = home_spread - away_spread
        
        has_impact = abs(net_total_impact) >= 2.0 or abs(net_spread_impact) >= 2.0
        
        if has_impact:
            logger.info(
                f"🏥 INJURIES: Total Impact={net_total_impact:+.1f}, "
                f"Spread Impact={net_spread_impact:+.1f}"
            )
        
        return {
            "has_impact": has_impact,
            "total_impact": net_total_impact,
            "spread_impact": net_spread_impact,
            "home_injuries": home_injuries.to_dict() if home_injuries else None,
            "away_injuries": away_injuries.to_dict() if away_injuries else None,
        }
    
    async def _get_narrative_signal(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Get public narrative signal"""
        
        narrative = await self.narrative_scraper.get_game_narrative(home_team, away_team)
        
        logger.info(
            f"📰 NARRATIVE: Public Lean={narrative['public_lean']:.0%}, "
            f"Strength={narrative['narrative_strength']:.0%}, "
            f"Vegas Bait={narrative['vegas_bait']}"
        )
        
        if narrative['storylines']:
            logger.info(f"   Storylines: {', '.join(narrative['storylines'])}")
        
        return narrative
    
    def _synthesize_signals(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        total: float,
        spread: float,
        line_signals: dict,
        weather_signal: dict,
        injury_signals: dict,
        narrative_signal: dict,
    ) -> Dict[str, Any]:
        """
        Synthesize all signals into unified recommendation.
        
        Signal Priority:
        1. Sharp line movement (highest confidence)
        2. Weather (physical constraint)
        3. Injury impact (known variable)
        4. Narrative/public lean (contrarian opportunity)
        """
        
        # Start with base line
        adjusted_total = total
        adjusted_spread = spread
        
        confidence = 0.5  # Base confidence
        edge_detected = False
        recommendations = []
        reasoning = []
        
        # 1. LINE MOVEMENT (highest priority - sharp money)
        if line_signals["sharp_detected"]:
            confidence += 0.25
            edge_detected = True
            
            best_move = line_signals["best_movement"]
            if best_move and best_move["market"] == "total":
                if best_move["move_direction"] == "up":
                    recommendations.append("FOLLOW_SHARP_OVER")
                    reasoning.append(f"Sharp money moved total UP by {abs(best_move['line_move']):.1f}")
                else:
                    recommendations.append("FOLLOW_SHARP_UNDER")
                    reasoning.append(f"Sharp money moved total DOWN by {abs(best_move['line_move']):.1f}")
        
        # 2. WEATHER (physical constraint)
        if weather_signal["has_impact"]:
            adjusted_total += weather_signal["adjustment"]
            confidence += 0.15 * weather_signal["severity"]
            
            if weather_signal["severity"] >= 0.5:
                edge_detected = True
                recommendations.append("WEATHER_UNDER")
                reasoning.append(f"Severe weather: {weather_signal['conditions']}")
        
        # 3. INJURIES (known variable)
        if injury_signals["has_impact"]:
            adjusted_total += injury_signals["total_impact"]
            adjusted_spread += injury_signals["spread_impact"]
            confidence += 0.1
            
            if abs(injury_signals["total_impact"]) >= 3.0:
                edge_detected = True
                if injury_signals["total_impact"] < 0:
                    recommendations.append("INJURY_UNDER")
                    reasoning.append(f"Key injuries impact: {injury_signals['total_impact']:+.1f} pts")
                else:
                    recommendations.append("INJURY_OVER")
                    reasoning.append(f"Defensive injuries boost offense: {injury_signals['total_impact']:+.1f} pts")
        
        # 4. NARRATIVE/PUBLIC (conspiracy and contrarian opportunity)
        public_lean = narrative_signal.get("public_lean", 0.5)
        narrative_strength = narrative_signal.get("narrative_strength", 0.5)
        conspiracy_score = narrative_signal.get("conspiracy_score", 0.0)
        betting_rec = narrative_signal.get("betting_recommendation", "NO_CLEAR_EDGE")
        sharp_vs_public = narrative_signal.get("sharp_vs_public", 0.5)
        
        # High conspiracy score boosts confidence
        if conspiracy_score > 0.7:
            confidence += 0.15
            edge_detected = True
            recommendations.append(betting_rec)
            reasoning.append(f"High conspiracy probability ({conspiracy_score:.0%})")
        
        # Contrarian fade opportunity
        elif public_lean > 0.75 and narrative_strength > 0.6:
            confidence += 0.1
            recommendations.append("FADE_PUBLIC_UNDER")
            reasoning.append(f"Fade public: {public_lean:.0%} on OVER")
        elif public_lean < 0.25 and narrative_strength > 0.6:
            confidence += 0.1
            recommendations.append("FADE_PUBLIC_OVER")
            reasoning.append(f"Fade public: {100-public_lean*100:.0%} on UNDER")
        
        # Sharp vs public divergence
        if abs(sharp_vs_public - 0.5) > 0.3:
            confidence += 0.05
            if sharp_vs_public > 0.7:
                recommendations.append("FOLLOW_SHARP")
                reasoning.append(f"Sharp money detected ({sharp_vs_public:.0%} sharp activity)")
        
        # Final recommendation
        if not recommendations:
            primary_recommendation = "PASS"
            reasoning.append("No strong edge detected")
        elif len(recommendations) == 1:
            primary_recommendation = recommendations[0]
        else:
            # Multiple signals - check consensus
            under_signals = sum(1 for r in recommendations if "UNDER" in r)
            over_signals = sum(1 for r in recommendations if "OVER" in r)
            
            if under_signals > over_signals:
                primary_recommendation = "STRONG_UNDER"
            elif over_signals > under_signals:
                primary_recommendation = "STRONG_OVER"
            else:
                primary_recommendation = "CONFLICTING_SIGNALS"
        
        # Calculate expected value
        total_adjustment = adjusted_total - total
        ev = abs(total_adjustment) * confidence
        
        # Confidence thresholds
        if confidence >= 0.8:
            bet_recommendation = "STRONG_BET"
        elif confidence >= 0.65:
            bet_recommendation = "LEAN"
        elif confidence >= 0.55:
            bet_recommendation = "SMALL_LEAN"
        else:
            bet_recommendation = "PASS"
        
        result = {
            "game_id": game_id,
            "matchup": f"{away_team} @ {home_team}",
            "market_lines": {
                "total": total,
                "spread": spread,
            },
            "adjusted_lines": {
                "total": adjusted_total,
                "spread": adjusted_spread,
            },
            "recommendation": primary_recommendation,
            "bet_strength": bet_recommendation,
            "confidence": confidence,
            "expected_value": ev,
            "edge_detected": edge_detected,
            "signals": {
                "line_movement": line_signals,
                "weather": weather_signal,
                "injuries": injury_signals,
                "narrative": narrative_signal,
            },
            "reasoning": reasoning,
        }
        
        # Log summary
        logger.info(f"\n🎯 RECOMMENDATION: {primary_recommendation}")
        logger.info(f"   Bet Strength: {bet_recommendation}")
        logger.info(f"   Confidence: {confidence:.0%}")
        logger.info(f"   Expected Value: {ev:.2f}")
        if reasoning:
            logger.info(f"   Reasoning:")
            for reason in reasoning:
                logger.info(f"      • {reason}")
        
        return result
    
    async def load_all_intelligence(self, odds_file: Optional[Path] = None):
        """Load all intelligence data sources"""
        
        logger.info("📡 Loading all intelligence systems...")
        
        # Find most recent odds file
        if not odds_file:
            odds_files = sorted(DATA_DIR.glob("nfl_odds_*.json"))
            if not odds_files:
                logger.error("No odds files found")
                return
            odds_file = odds_files[-1]
        
        logger.info(f"   Odds: {odds_file.name}")
        
        # Load line movements
        self.line_tracker.ingest_odds_file(odds_file)
        logger.info(f"   ✅ Line movement tracker loaded")
        
        # Load weather
        weather_list = await fetch_weather_for_games(odds_file)
        self.weather_reports = {w.team: w for w in weather_list}
        logger.info(f"   ✅ Weather data loaded ({len(weather_list)} teams)")
        
        # Load injuries
        self.injury_reports = await fetch_injuries_for_games(odds_file)
        logger.info(f"   ✅ Injury reports loaded ({len(self.injury_reports)} teams)")
        
        logger.info("✅ All intelligence systems ready\n")
    
    async def analyze_all_games(self, odds_file: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Analyze all games with unified intelligence"""
        
        # Load intelligence
        await self.load_all_intelligence(odds_file)
        
        # Find odds file
        if not odds_file:
            odds_files = sorted(DATA_DIR.glob("nfl_odds_*.json"))
            if not odds_files:
                return []
            odds_file = odds_files[-1]
        
        # Load games
        odds_data = json.loads(odds_file.read_text())
        
        # Analyze each game
        recommendations = []
        
        for game_odds in odds_data:
            game_id = game_odds.get("game_id")
            home_team = game_odds.get("home_team")
            away_team = game_odds.get("away_team")
            total = game_odds.get("total")
            spread = game_odds.get("spread_home")
            
            if not all([game_id, home_team, away_team, total]):
                continue
            
            try:
                recommendation = await self.analyze_game(
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    total=total,
                    spread=spread or 0.0,
                )
                
                recommendations.append(recommendation)
            
            except Exception as e:
                logger.error(f"Failed to analyze {game_id}: {e}")
        
        # Save unified analysis
        output_file = DATA_DIR / f"unified_analysis_{date.today()}.json"
        output_file.write_text(json.dumps(recommendations, indent=2))
        logger.info(f"\n💾 Saved unified analysis to {output_file}")
        
        return recommendations


async def main():
    """Main unified intelligence runner"""
    
    intelligence = UnifiedBettingIntelligence()
    recommendations = await intelligence.analyze_all_games()
    
    # Print summary
    print("\n" + "=" * 80)
    print("🧠 UNIFIED BETTING INTELLIGENCE REPORT")
    print("=" * 80)
    
    # Filter strong bets
    strong_bets = [r for r in recommendations if r["bet_strength"] in ["STRONG_BET", "LEAN"]]
    
    if strong_bets:
        print(f"\n🔥 {len(strong_bets)} STRONG BETTING OPPORTUNITIES:\n")
        
        for rec in sorted(strong_bets, key=lambda x: x["confidence"], reverse=True):
            print(f"{'='*80}")
            print(f"🎯 {rec['matchup']}")
            print(f"   Recommendation: {rec['recommendation']} ({rec['bet_strength']})")
            print(f"   Confidence: {rec['confidence']:.0%} | EV: {rec['expected_value']:.2f}")
            print(f"   Market Total: {rec['market_lines']['total']}")
            print(f"   Adjusted Total: {rec['adjusted_lines']['total']:.1f}")
            print(f"\n   🔍 Key Factors:")
            for reason in rec['reasoning']:
                print(f"      • {reason}")
            print()
    else:
        print("\n⏳ No strong betting opportunities detected today")
    
    print("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )
    asyncio.run(main())
