#!/usr/bin/env python3
"""
Unified Betting Intelligence: Narratives + Sentiment + AI Council + Referees
=============================================================================

Combines:
1. Media narratives (trap games, revenge games, first Monday night in 7 years)
2. Public sentiment (Reddit, expert picks, sharp vs. public)
3. AI Council predictions (7 models with referee features)
4. Referee bias patterns (22 refs, overtime specialists, home advantages)

Creates actionable betting recommendations with narrative context.
"""

import asyncio
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Literal
from datetime import datetime
from pathlib import Path
from io import BytesIO
import joblib

from betting_types import (
    NarrativeType,
    BetSize,
    RiskLevel,
    SpreadPrediction,
    TotalPrediction,
    MoneylinePrediction,
    RefereeProfile,
    GameData,
    NarrativeContext,
    SentimentContext,
    UnifiedPrediction,
)
from config_loader import BettingConfig


class SecurityError(RuntimeError):
    """Raised when a model artifact fails checksum validation."""


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds >= 0:
        return 1 + (odds / 100.0)
    return 1 + (100.0 / abs(odds))


def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability in the range [0, 1]."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def remove_vig(home_prob: float, away_prob: float) -> Tuple[float, float]:
    """Normalize two probabilities so they sum to one (vig removal)."""
    total = home_prob + away_prob
    if total == 0:
        return 0.5, 0.5
    return home_prob / total, away_prob / total


class NarrativeIntegratedAICouncil:
    """AI Council that incorporates narratives into betting decisions"""
    
    VERSION = "1.2.0"

    def __init__(self, models_dir: str = 'models', config: Optional[BettingConfig] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or BettingConfig.from_yaml()
        self.expected_model_hashes = self.config.model_checksums or {}
        self.models_dir = Path(models_dir)
        self.models = {}
        self.referee_data = {}
        self.narrative_scraper = None
        self.degraded_mode = False
        self.degraded_reasons: List[str] = []
        self.load_models()
    
    def load_models(self):
        """Load trained AI Council models"""
        
        model_files = [
            'spread_expert_nfl_model.pkl',
            'contrarian_nfl_model.pkl',
            'home_advantage_nfl_model.pkl',
            'total_regressor_nfl_model.pkl',
            'total_high_games_nfl_model.pkl',
            'total_low_games_nfl_model.pkl',
            'total_weather_adjusted_nfl_model.pkl'
        ]
        
        for model_file in model_files:
            path = self.models_dir / model_file
            model_key = model_file.replace('_nfl_model.pkl', '')
            expected_hash = self.expected_model_hashes.get(model_file)
            try:
                self.models[model_key] = self._load_model_secure(path, expected_hash)
            except FileNotFoundError as exc:
                self.logger.error("Model file not found: %s", exc, exc_info=True)
                self.degraded_mode = True
                self.degraded_reasons.append(f"missing model {model_file}")
            except SecurityError as exc:
                self.logger.error(str(exc), exc_info=True)
                raise
            except Exception as exc:
                self.logger.error("Unexpected error loading %s: %s", path, exc, exc_info=True)
                self.degraded_mode = True
                self.degraded_reasons.append(f"error loading {model_file}")
        
        # Load referee data
        ref_path = Path('data/referee_training_features.json')
        if ref_path.exists():
            with open(ref_path, 'r') as f:
                data = json.load(f)
                self.referee_data = data.get('referee_profiles', {})
        
        self.logger.info("Loaded %d models and %d referee profiles", len(self.models), len(self.referee_data))

    def _load_model_secure(self, filepath: Path, expected_hash: Optional[str] = None):
        """Load a model artifact with optional checksum verification."""
        if not filepath.exists():
            raise FileNotFoundError(str(filepath))
        with filepath.open('rb') as fh:
            content = fh.read()
        if expected_hash:
            actual_hash = hashlib.sha256(content).hexdigest()
            if actual_hash != expected_hash:
                raise SecurityError(f"Model hash mismatch for {filepath}: expected {expected_hash}, got {actual_hash}")
        return joblib.load(BytesIO(content))
    
    def extract_narrative(self, game_data: GameData) -> NarrativeContext:
        """
        Extract media narrative for a game
        
        Looks for:
        - Historical matchups (first game in X years)
        - Revenge games (recent history)
        - Trap games (big spread after big win)
        - Storylines (coaching changes, QB debuts)
        - Revenge narratives
        """
        
        narratives: List[NarrativeContext] = []
        
        days_since_mnf = game_data.get('days_since_last_monday_night')
        if days_since_mnf and days_since_mnf > 365 * 5:
            narratives.append(
                NarrativeContext(
                    storyline=f"First Monday night in {days_since_mnf // 365} years",
                    type=NarrativeType.HOMECOMING,
                    strength=0.7,
                    impact_home_team=-0.03,
                    expected_performance_delta=-0.09,
                    confidence=0.8,
                )
            )
        
        last_meeting = game_data.get('last_meeting_result')
        if last_meeting:
            narratives.append(
                NarrativeContext(
                    storyline=f"Rematch after {last_meeting}",
                    type=NarrativeType.REVENGE_GAME,
                    strength=0.6,
                    impact_home_team=0.02,
                    expected_performance_delta=0.06,
                    confidence=0.7,
                )
            )
        
        if game_data.get('is_trap_game'):
            narratives.append(
                NarrativeContext(
                    storyline='Potential trap after big win',
                    type=NarrativeType.TRAP_GAME,
                    strength=0.8,
                    impact_home_team=-0.05,
                    expected_performance_delta=-0.15,
                    confidence=0.75,
                )
            )

        if game_data.get('injuries') or game_data.get('injury_report'):
            narratives.append(
                NarrativeContext(
                    storyline='Injury revenge angle for returning starters',
                    type=NarrativeType.INJURY_REVENGE,
                    strength=0.7,
                    impact_home_team=0.05,
                    expected_performance_delta=0.15,
                    confidence=0.75,
                )
            )
        
        if narratives:
            return max(narratives, key=lambda ctx: ctx.strength)
        
        return NarrativeContext(
            storyline="Standard matchup",
            type=NarrativeType.NEUTRAL,
            strength=0.0,
            impact_home_team=0.0,
            expected_performance_delta=0.0,
            confidence=0.5
        )
    
    def extract_referee_context(self, referee_name: str, home_team: str = None) -> RefereeProfile:
        """Get referee profile and impact"""
        
        if referee_name in self.referee_data:
            profile = self.referee_data[referee_name]
            overtime_frequency = profile.get('avg_overtime_rate', 0.06)
            if overtime_frequency > 1:
                overtime_frequency /= 100.0
            overtime_frequency = max(0.0, min(1.0, overtime_frequency))
            games = int(profile.get('games', 0) or 0)

            return RefereeProfile(
                name=referee_name,
                avg_margin=profile.get('avg_margin', 0.0),
                penalty_rate=profile.get('avg_penalties', 6.1),
                overtime_frequency=overtime_frequency,
                home_advantage=profile.get('avg_margin', 0.0),
                classification=', '.join(profile.get('labels', [])),
                games_reffed=games,
            )
        
        # Default neutral referee
        return RefereeProfile(
            name=referee_name,
            avg_margin=0.0,
            penalty_rate=6.1,
            overtime_frequency=0.06,
            home_advantage=0.0,
            classification="unknown",
            games_reffed=0,
        )
    
    def make_unified_prediction(self, game_data: GameData) -> UnifiedPrediction:
        """Generate a unified prediction using typed inputs."""

        self.degraded_mode = False
        self.degraded_reasons.clear()

        ts = game_data.get('timestamp')
        if isinstance(ts, datetime):
            timestamp = ts
        elif isinstance(ts, str):
            try:
                timestamp = datetime.fromisoformat(ts)
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        narrative = self.extract_narrative(game_data)
        sentiment = self._extract_sentiment(game_data)
        referee = self.extract_referee_context(
            game_data.get('referee', 'Unknown'),
            game_data.get('home_team'),
        )

        spread_pred = self._get_spread_prediction(game_data, narrative, referee)
        total_pred = self._get_total_prediction(game_data, narrative, referee)
        moneyline_pred = self._get_ml_prediction(game_data, narrative, referee)

        recommendation = self._generate_unified_recommendation(
            spread_pred, total_pred, moneyline_pred, narrative, sentiment, referee
        )
        edge_signals = self._identify_edge_signals(narrative, sentiment, referee)
        confidence = self._calculate_confidence(spread_pred, sentiment, referee)
        risk_level = self._assess_risk_level(narrative, sentiment, edge_signals)

        prediction = UnifiedPrediction(
            game_id=game_data.get('game_id', 'UNKNOWN'),
            home_team=game_data.get('home_team', 'HOME'),
            away_team=game_data.get('away_team', 'AWAY'),
            timestamp=timestamp,
            spread_prediction=spread_pred,
            total_prediction=total_pred,
            moneyline_prediction=moneyline_pred,
            narrative=narrative,
            sentiment=sentiment,
            referee=referee,
            recommendation=recommendation,
            confidence=confidence,
            risk_level=risk_level,
            edge_signals=edge_signals,
            degraded=self.degraded_mode,
            degraded_reasons=list(self.degraded_reasons),
            version=self.VERSION,
        )
        return prediction
    
    def _extract_sentiment(self, game_data: GameData) -> SentimentContext:
        """Extract sentiment (simplified placeholder for real integration)."""

        reddit = float(game_data.get('reddit_lean', 0.0) or 0.0)
        expert = float(game_data.get('expert_pct_home', 0.5) or 0.5)
        sharp_ml = float(game_data.get('sharp_public_ml', 0.0) or 0.0)
        sharp_total = float(game_data.get('sharp_public_total', 0.0) or 0.0)
        contrarian = float(game_data.get('contrarian_opportunity', 0.0) or 0.0)
        crowd_roar = float(game_data.get('crowd_roar', 0.0) or 0.0)

        return SentimentContext(
            reddit_lean=reddit,
            expert_consensus=expert,
            sharp_vs_public_ml=sharp_ml,
            sharp_vs_public_total=sharp_total,
            contrarian_score=contrarian,
            crowd_roar_signal=crowd_roar,
        )
    
    def _get_spread_prediction(self, game_data: GameData, narrative: NarrativeContext, referee: RefereeProfile) -> SpreadPrediction:
        """Get spread prediction with narrative adjustment"""
        
        home_pct = game_data.get('spread_model_home_pct', 0.5) or 0.5
        pick: Literal["home", "away"] = 'home' if home_pct > 0.5 else 'away'
        base_confidence = max(0.0, min(1.0, abs(home_pct - 0.5) * 2))
        line = float(game_data.get('spread', 0.0))
        american_odds = int(game_data.get('spread_odds', -110))
        
        narrative_adjustment = narrative.expected_performance_delta
        referee_adjustment = referee.avg_margin * 0.5
        total_adjustment = narrative_adjustment + referee_adjustment
        adjusted_line = line + total_adjustment
        edge = total_adjustment

        return SpreadPrediction(
            pick=pick,
            adjusted_line=adjusted_line,
            confidence=base_confidence,
            edge=edge,
            american_odds=american_odds,
        )

    def _get_total_prediction(self, game_data: GameData, narrative: NarrativeContext, referee: RefereeProfile) -> TotalPrediction:
        """Get total prediction with narrative adjustment"""
        
        over_pct = game_data.get('total_model_over_pct', 0.5) or 0.5
        pick: Literal["over", "under"] = 'over' if over_pct > 0.5 else 'under'
        confidence = max(0.0, min(1.0, abs(over_pct - 0.5) * 2))
        line = float(game_data.get('total', 44.5))
        american_odds = int(game_data.get('total_odds', -110))
        
        narrative_scoring_impact = -0.5 if narrative.type == NarrativeType.TRAP_GAME else 0.0
        ref_scoring_impact = 0.5 if referee.overtime_frequency > self.config.thresholds['referee']['ot_specialist_threshold'] else 0.0
        total_adjustment = narrative_scoring_impact + ref_scoring_impact
        adjusted_line = line + total_adjustment

        return TotalPrediction(
            pick=pick,
            adjusted_line=adjusted_line,
            confidence=confidence,
            edge=total_adjustment,
            american_odds=american_odds,
        )

    def _get_ml_prediction(self, game_data: GameData, narrative: NarrativeContext, referee: RefereeProfile) -> MoneylinePrediction:
        """Get moneyline prediction"""
        
        home_odds = int(game_data.get('home_ml_odds', -110))
        away_odds = int(game_data.get('away_ml_odds', 110))
        implied_home = american_to_implied_prob(home_odds)
        implied_away = american_to_implied_prob(away_odds)
        implied_home, implied_away = remove_vig(implied_home, implied_away)

        model_bias = game_data.get('home_advantage_pct')
        if model_bias is not None:
            pick = 'home' if model_bias > 0.5 else 'away'
            model_confidence = abs(model_bias - 0.5) * 2
        else:
            pick = 'home' if implied_home >= implied_away else 'away'
            model_confidence = abs(implied_home - implied_away)

        primary_prob = implied_home if pick == 'home' else implied_away
        confidence = max(0.0, min(1.0, max(model_confidence, primary_prob)))
        edge = primary_prob - 0.5

        return MoneylinePrediction(
            pick=pick,
            confidence=confidence,
            edge=edge,
            home_odds=home_odds,
            away_odds=away_odds,
        )
    
    def _generate_unified_recommendation(
        self,
        spread: SpreadPrediction,
        total: TotalPrediction,
        ml: MoneylinePrediction,
        narrative: NarrativeContext,
        sentiment: SentimentContext,
        referee: RefereeProfile,
    ) -> Dict[str, Any]:
        """Generate final betting recommendation"""
        
        recommendation = {
            'primary_play': None,
            'size': BetSize.SMALL.value,
            'reasoning': [],
            'secondary_plays': []
        }

        contrarian_conf = self.config.thresholds['contrarian']
        
        # Contrarian signal (strongest)
        if sentiment.contrarian_score > contrarian_conf['trigger']:
            recommendation['primary_play'] = "CONTRARIAN: Fade public"
            recommendation['size'] = (
                BetSize.LARGE.value if sentiment.contrarian_score > contrarian_conf['strong_trigger'] else BetSize.MEDIUM.value
            )
            recommendation['reasoning'].append(
                f"Sharp/public divergence {sentiment.contrarian_score:.1%} - public heavily biased"
            )
        
        # Narrative trap game signal
        elif (
            narrative.type == NarrativeType.TRAP_GAME
            and narrative.strength > self.config.thresholds['narrative']['trap_strength_min']
        ):
            recommendation['primary_play'] = f"FADE {narrative.storyline}"
            recommendation['size'] = BetSize.MEDIUM.value
            recommendation['reasoning'].append(
                f"Narrative trap game - typical letdown spot ({narrative.impact_home_team:+.2f} edge)"
            )
        
        # Crowd roar signal
        elif sentiment.crowd_roar_signal > 0.6:
            recommendation['primary_play'] = "CROWD ROAR EDGE: Adjust totals up +5%"
            recommendation['size'] = BetSize.MEDIUM.value
            recommendation['reasoning'].append(
                "League letting them play - expect increased scoring next drive"
            )
        
        # Referee special (OT specialist)
        if referee.overtime_frequency > self.config.thresholds['referee']['ot_specialist_threshold']:
            recommendation['secondary_plays'].append({
                'play': 'OVERTIME PROPS UP +1.5%',
                'reason': f"{referee.name} has {referee.overtime_frequency:.1%} OT rate"
            })
        
        # Add narrative context
        if narrative.strength > 0.5:
            recommendation['reasoning'].append(f"Narrative factor: {narrative.storyline}")
        
        return recommendation
    
    def _identify_edge_signals(
        self,
        narrative: NarrativeContext,
        sentiment: SentimentContext,
        referee: RefereeProfile,
    ) -> List[str]:
        """Identify specific edge signals"""
        
        signals = []
        
        if sentiment.contrarian_score > self.config.thresholds['contrarian']['trigger']:
            signals.append(f"CONTRARIAN_EDGE_{sentiment.contrarian_score:.0%}")
        
        if (
            narrative.type == NarrativeType.TRAP_GAME
            and narrative.strength > self.config.thresholds['narrative']['trap_strength_min']
        ):
            signals.append("NARRATIVE_TRAP_GAME")
        
        if sentiment.crowd_roar_signal > 0.6:
            signals.append("CROWD_ROAR_SIGNAL")
        
        if referee.overtime_frequency > self.config.thresholds['referee']['ot_specialist_threshold']:
            signals.append(f"OT_SPECIALIST_REF_{referee.overtime_frequency:.0%}")
        
        if narrative.type == NarrativeType.HOMECOMING:
            signals.append("HOMECOMING_NARRATIVE")
        if narrative.type == NarrativeType.INJURY_REVENGE:
            signals.append("INJURY_REVENGE_NARRATIVE")
        
        if sentiment.sharp_vs_public_ml > 0.3:
            signals.append("SHARP_PUBLIC_DIVERGENCE")
        
        return signals
    
    def _calculate_confidence(
        self,
        spread: SpreadPrediction,
        sentiment: SentimentContext,
        referee: RefereeProfile,
    ) -> float:
        """Calculate overall confidence in recommendation"""
        
        base_confidence = 0.5
        
        # Model confidence
        base_confidence += spread.confidence * 0.3
        
        # Sentiment agreement boost
        base_confidence += min(0.2, sentiment.contrarian_score * 0.3)
        
        # Referee clarity boost (clear specialist types)
        if referee.classification:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

    def _assess_risk_level(
        self,
        narrative: NarrativeContext,
        sentiment: SentimentContext,
        edge_signals: List[str],
    ) -> RiskLevel:
        """Assess risk level of recommendation"""
        
        if len(edge_signals) > 3:
            return RiskLevel.LOW
        if len(edge_signals) > 1:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH


def format_prediction_for_output(prediction: UnifiedPrediction) -> str:
    """Format prediction for display"""
    
    output = f"""
╔══════════════════════════════════════════════════════════════════╗
║  {prediction.away_team:>10} @ {prediction.home_team:<10} - {prediction.referee.name}
╚══════════════════════════════════════════════════════════════════╝

📊 PREDICTIONS:
   Spread: {prediction.spread_prediction.pick:>6}  (Line: {prediction.spread_prediction.adjusted_line:+.1f})
   Total:  {prediction.total_prediction.pick:>6}  (Line: {prediction.total_prediction.adjusted_line:.1f})
   ML:     {prediction.moneyline_prediction.pick:>6}  

🎭 NARRATIVES:
   Story: {prediction.narrative.storyline}
   Type:  {prediction.narrative.type.value}
   Edge:  {prediction.narrative.expected_performance_delta:+.2f} points

💬 SENTIMENT:
   Reddit:    {prediction.sentiment.reddit_lean:+.2f}
   Contrarian: {prediction.sentiment.contrarian_score:.0%}
   Crowd Roar: {prediction.sentiment.crowd_roar_signal:.0%}

🏈 REFEREE:
   {prediction.referee.name}
   Margin: {prediction.referee.avg_margin:+.2f}
   OT Rate: {prediction.referee.overtime_frequency:.1%}
   Type: {prediction.referee.classification}

🎯 RECOMMENDATION:
   {prediction.recommendation['primary_play']}
   Size: {prediction.recommendation['size'].upper()}
   Risk: {prediction.risk_level.value.upper()}
   
   Reasoning:
"""
    
    for reason in prediction.recommendation['reasoning']:
        output += f"   • {reason}\n"
    
    if prediction.recommendation['secondary_plays']:
        output += f"\n   Secondary Plays:\n"
        for play in prediction.recommendation['secondary_plays']:
            output += f"   • {play['play']}: {play['reason']}\n"
    
    output += f"\n   Edge Signals: {', '.join(prediction.edge_signals)}\n"
    output += f"   Confidence: {prediction.confidence:.0%}\n"
    
    return output


def main():
    """Example usage"""
    
    print("🏈 UNIFIED AI COUNCIL: NARRATIVES + SENTIMENT + REFEREES")
    print("=" * 60)
    
    council = NarrativeIntegratedAICouncil()
    
    # Example game with narrative
    game_data = {
        'game_id': 'ATL_vs_KC_2025_monday',
        'home_team': 'Chiefs',
        'away_team': 'Falcons',
        'referee': 'Bill Vinovich',
        'spread': -7.0,
        'total': 44.5,
        'home_ml_odds': 1.65,
        'away_ml_odds': 2.20,
        'days_since_last_monday_night': 365 * 7,  # First Monday in 7 years
        'is_trap_game': False,
        'reddit_lean': 0.3,  # Lean under
        'expert_pct_home': 0.65,
        'contrarian_opportunity': 0.75,  # Strong contrarian signal
        'crowd_roar': 0.0,
        'spread_model_home_pct': 0.58,
        'total_model_over_pct': 0.45,
        'home_advantage_pct': 0.62,
    }
    
    prediction = council.make_unified_prediction(game_data)
    print(format_prediction_for_output(prediction))


if __name__ == "__main__":
    main()
