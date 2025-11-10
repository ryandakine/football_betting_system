#!/usr/bin/env python3
"""
Intelligent Model Selector
==========================
Automatically selects the best prediction model based on available data.

Decision Tree:
1. Strong referee edge exists? → Use Referee Intelligence Model
2. Narrative/storyline exists? → Use Narrative Model + ML ensemble
3. Historical matchup data? → Use XGBoost/Neural Net
4. Sharp money divergence? → Use Market Analysis Model
5. Default → Use Full Ensemble (all models weighted)

Usage:
    python intelligent_model_selector.py --week 10 --game "PIT @ LAC"
    python intelligent_model_selector.py --week 10 --all-games
"""

import argparse
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import sentiment analysis
try:
    from ai_council_with_sentiment import SentimentFeatureExtractor
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False


class ModelType(Enum):
    """Available prediction models."""
    REFEREE_INTELLIGENCE = "referee_intelligence"
    NARRATIVE = "narrative"
    PUBLIC_SENTIMENT_CONTRARIAN = "public_sentiment_contrarian"  # NEW
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    MARKET_ANALYSIS = "market_analysis"
    ENSEMBLE = "ensemble"
    SITUATIONAL = "situational"


@dataclass
class ModelSelection:
    """Result of model selection for a game."""
    game: str
    primary_model: ModelType
    confidence_score: float
    reasoning: str
    fallback_models: List[ModelType]
    data_availability: Dict[str, bool]
    narrative_strength: float
    referee_edge_strength: float
    # Sentiment data
    contrarian_score: float = 0.0
    public_sentiment: float = 0.0
    sharp_divergence: float = 0.0


class IntelligentModelSelector:
    """Selects optimal prediction model for each game."""

    def __init__(self):
        # Initialize sentiment extractor if available
        self.sentiment_extractor = SentimentFeatureExtractor() if SENTIMENT_AVAILABLE else None

        # Model priority weights
        self.model_priorities = {
            ModelType.REFEREE_INTELLIGENCE: {
                'min_edge_strength': 0.65,  # 65% confidence
                'weight': 1.0
            },
            ModelType.PUBLIC_SENTIMENT_CONTRARIAN: {
                'min_contrarian_score': 0.70,  # 70% contrarian signal
                'weight': 0.95  # Very high - this is +EV
            },
            ModelType.NARRATIVE: {
                'min_narrative_strength': 0.60,
                'weight': 0.9
            },
            ModelType.XGBOOST: {
                'min_historical_games': 10,
                'weight': 0.85
            },
            ModelType.NEURAL_NETWORK: {
                'min_training_samples': 100,
                'weight': 0.85
            },
            ModelType.MARKET_ANALYSIS: {
                'min_sharp_divergence': 2.0,  # 2 points
                'weight': 0.80
            },
            ModelType.SITUATIONAL: {
                'min_factors': 3,
                'weight': 0.75
            },
            ModelType.ENSEMBLE: {
                'weight': 0.70  # Default fallback
            }
        }

    def select_model(
        self,
        game: str,
        referee_edge: Optional[float] = None,
        narrative_strength: Optional[float] = None,
        historical_games: int = 0,
        sharp_divergence: Optional[float] = None,
        situational_factors: int = 0,
        # Sentiment data
        game_data: Optional[Dict] = None,  # Full game data for sentiment extraction
        contrarian_score: Optional[float] = None,
        public_sentiment: Optional[float] = None
    ) -> ModelSelection:
        """
        Select best model for a game based on available data.

        Args:
            game: Game matchup (e.g., "PIT @ LAC")
            referee_edge: Referee intelligence confidence (0-1)
            narrative_strength: Narrative storyline strength (0-1)
            historical_games: Number of historical matchups
            sharp_divergence: Sharp vs public money difference (points)
            situational_factors: Count of situational factors (weather, injuries, etc.)
            game_data: Full game data dict for sentiment extraction
            contrarian_score: Contrarian opportunity score (0-1)
            public_sentiment: Public sentiment score (-1 to +1)

        Returns:
            ModelSelection with recommended model and reasoning
        """

        # Extract sentiment if game_data provided and extractor available
        sentiment_data = {}
        if game_data and self.sentiment_extractor:
            sentiment_data = self.sentiment_extractor.extract_game_sentiment(game_data)
            # Override with extracted values if not explicitly provided
            if contrarian_score is None:
                contrarian_score = sentiment_data.get('contrarian_opportunity', 0.0)
            if public_sentiment is None:
                public_sentiment = sentiment_data.get('reddit_sentiment_score', 0.0)

        data_availability = {
            'referee_edge': referee_edge is not None and referee_edge > 0,
            'public_sentiment': contrarian_score is not None and contrarian_score > 0,
            'narrative': narrative_strength is not None and narrative_strength > 0,
            'historical_data': historical_games >= 10,
            'market_data': sharp_divergence is not None,
            'situational_factors': situational_factors > 0
        }

        # Decision tree
        selected_model = None
        confidence = 0.0
        reasoning = ""
        fallbacks = []

        # Priority 1: Strong referee edge
        if referee_edge and referee_edge >= self.model_priorities[ModelType.REFEREE_INTELLIGENCE]['min_edge_strength']:
            selected_model = ModelType.REFEREE_INTELLIGENCE
            confidence = referee_edge
            reasoning = f"Strong referee edge detected ({referee_edge:.1%} confidence). Referee intelligence is the primary edge."
            fallbacks = [ModelType.ENSEMBLE]

        # Priority 2: Strong contrarian sentiment (public heavily on one side, sharp disagrees)
        elif contrarian_score and contrarian_score >= self.model_priorities[ModelType.PUBLIC_SENTIMENT_CONTRARIAN]['min_contrarian_score']:
            selected_model = ModelType.PUBLIC_SENTIMENT_CONTRARIAN
            confidence = contrarian_score
            reasoning = f"Strong contrarian opportunity detected ({contrarian_score:.1%} score). Public heavily on one side, sharp money disagrees. This is a +EV edge."
            fallbacks = [ModelType.ENSEMBLE]

        # Priority 3: Strong narrative + no referee edge or contrarian signal
        elif narrative_strength and narrative_strength >= self.model_priorities[ModelType.NARRATIVE]['min_narrative_strength']:
            selected_model = ModelType.NARRATIVE
            confidence = narrative_strength
            reasoning = f"Compelling narrative detected ({narrative_strength:.1%} strength). No referee edge available, using narrative + ML ensemble."
            fallbacks = [ModelType.XGBOOST, ModelType.NEURAL_NETWORK, ModelType.ENSEMBLE]

        # Priority 3: Sufficient historical data for ML
        elif historical_games >= self.model_priorities[ModelType.XGBOOST]['min_historical_games']:
            selected_model = ModelType.XGBOOST
            confidence = min(0.70, 0.50 + (historical_games / 100))  # Scale with data
            reasoning = f"Sufficient historical matchup data ({historical_games} games). Using XGBoost classifier."
            fallbacks = [ModelType.NEURAL_NETWORK, ModelType.ENSEMBLE]

        # Priority 4: Market analysis (sharp divergence)
        elif sharp_divergence and abs(sharp_divergence) >= self.model_priorities[ModelType.MARKET_ANALYSIS]['min_sharp_divergence']:
            selected_model = ModelType.MARKET_ANALYSIS
            confidence = min(0.65, 0.50 + (abs(sharp_divergence) / 10))
            reasoning = f"Sharp money divergence detected ({sharp_divergence:+.1f} points). Market inefficiency opportunity."
            fallbacks = [ModelType.ENSEMBLE]

        # Priority 5: Situational factors
        elif situational_factors >= self.model_priorities[ModelType.SITUATIONAL]['min_factors']:
            selected_model = ModelType.SITUATIONAL
            confidence = min(0.60, 0.45 + (situational_factors * 0.05))
            reasoning = f"Multiple situational factors present ({situational_factors}). Using situational specialist model."
            fallbacks = [ModelType.ENSEMBLE]

        # Default: Full ensemble
        else:
            selected_model = ModelType.ENSEMBLE
            confidence = 0.55  # Baseline ensemble confidence
            reasoning = "No single model has strong edge. Using weighted ensemble of all models."
            fallbacks = []

        return ModelSelection(
            game=game,
            primary_model=selected_model,
            confidence_score=confidence,
            reasoning=reasoning,
            fallback_models=fallbacks,
            data_availability=data_availability,
            narrative_strength=narrative_strength or 0.0,
            referee_edge_strength=referee_edge or 0.0,
            contrarian_score=contrarian_score or 0.0,
            public_sentiment=public_sentiment or 0.0,
            sharp_divergence=sentiment_data.get('sharp_public_split_ml', 0.0) if sentiment_data else 0.0
        )

    def generate_report(self, selections: List[ModelSelection]) -> str:
        """Generate model selection report."""
        report = []
        report.append("=" * 80)
        report.append("🤖 INTELLIGENT MODEL SELECTION REPORT")
        report.append("=" * 80)
        report.append("")

        # Group by model type
        by_model = {}
        for sel in selections:
            model_name = sel.primary_model.value
            if model_name not in by_model:
                by_model[model_name] = []
            by_model[model_name].append(sel)

        # Summary
        report.append("📊 MODEL DISTRIBUTION:")
        for model, games in sorted(by_model.items(), key=lambda x: len(x[1]), reverse=True):
            report.append(f"   {model.upper()}: {len(games)} games")
        report.append("")

        # Per-game details
        report.append("=" * 80)
        report.append("🏈 GAME-BY-GAME MODEL SELECTION")
        report.append("=" * 80)
        report.append("")

        for sel in selections:
            report.append(f"Game: {sel.game}")
            report.append(f"  ✅ PRIMARY MODEL: {sel.primary_model.value.upper()}")
            report.append(f"  📊 Confidence: {sel.confidence_score:.1%}")
            report.append(f"  💡 Reasoning: {sel.reasoning}")

            if sel.fallback_models:
                fallback_names = [m.value for m in sel.fallback_models]
                report.append(f"  🔄 Fallbacks: {', '.join(fallback_names)}")

            report.append("")
            report.append(f"  📁 Data Availability:")
            for data_type, available in sel.data_availability.items():
                status = "✅" if available else "❌"
                report.append(f"     {status} {data_type}")

            if sel.referee_edge_strength > 0:
                report.append(f"  🔍 Referee Edge: {sel.referee_edge_strength:.1%}")
            if sel.narrative_strength > 0:
                report.append(f"  📖 Narrative Strength: {sel.narrative_strength:.1%}")
            if sel.contrarian_score > 0:
                report.append(f"  🎯 Contrarian Opportunity: {sel.contrarian_score:.1%}")
            if sel.public_sentiment != 0:
                sentiment_emoji = "📈" if sel.public_sentiment > 0 else "📉"
                report.append(f"  {sentiment_emoji} Public Sentiment: {sel.public_sentiment:+.2f}")
            if sel.sharp_divergence != 0:
                report.append(f"  💰 Sharp/Public Divergence: {sel.sharp_divergence:+.2f}")

            report.append("")
            report.append("-" * 80)
            report.append("")

        report.append("=" * 80)
        report.append("💡 RECOMMENDATIONS")
        report.append("=" * 80)
        report.append("")

        # Count games with strong edges
        strong_edges = [s for s in selections if s.confidence_score >= 0.70]
        report.append(f"✅ {len(strong_edges)} games with strong edges (70%+ confidence)")

        # Count games needing ensemble
        ensemble_games = [s for s in selections if s.primary_model == ModelType.ENSEMBLE]
        if ensemble_games:
            report.append(f"⚠️  {len(ensemble_games)} games require ensemble approach (no clear edge)")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def example_usage():
    """Example of intelligent model selection."""
    selector = IntelligentModelSelector()

    # Example 1: Strong referee edge
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Strong Referee Edge")
    print("=" * 80)

    selection1 = selector.select_model(
        game="BUF @ KC",
        referee_edge=0.82,  # 82% confidence from Brad Rogers bias
        narrative_strength=0.30,
        historical_games=15
    )
    print(f"\nGame: {selection1.game}")
    print(f"Selected Model: {selection1.primary_model.value}")
    print(f"Confidence: {selection1.confidence_score:.1%}")
    print(f"Reasoning: {selection1.reasoning}")

    # Example 2: No referee edge, strong narrative
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Strong Narrative (Neutral Referee)")
    print("=" * 80)

    selection2 = selector.select_model(
        game="PIT @ LAC",
        referee_edge=0.52,  # Neutral referee (Clay Martin)
        narrative_strength=0.75,  # Strong storyline
        historical_games=8
    )
    print(f"\nGame: {selection2.game}")
    print(f"Selected Model: {selection2.primary_model.value}")
    print(f"Confidence: {selection2.confidence_score:.1%}")
    print(f"Reasoning: {selection2.reasoning}")

    # Example 3: Strong contrarian sentiment
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Contrarian Sentiment Edge")
    print("=" * 80)

    selection3 = selector.select_model(
        game="DET @ GB",
        referee_edge=0.48,  # Weak referee edge
        narrative_strength=0.40,  # Weak narrative
        historical_games=5,
        contrarian_score=0.82,  # 82% contrarian signal (strong!)
        public_sentiment=-0.65  # Public heavily on one side
    )
    print(f"\nGame: {selection3.game}")
    print(f"Selected Model: {selection3.primary_model.value}")
    print(f"Confidence: {selection3.confidence_score:.1%}")
    print(f"Reasoning: {selection3.reasoning}")
    print(f"Contrarian Score: {selection3.contrarian_score:.1%}")

    # Example 4: No edges, default to ensemble
    print("\n" + "=" * 80)
    print("EXAMPLE 4: No Clear Edge (Ensemble)")
    print("=" * 80)

    selection4 = selector.select_model(
        game="ATL @ CAR",
        referee_edge=0.48,
        narrative_strength=0.40,
        historical_games=5
    )
    print(f"\nGame: {selection4.game}")
    print(f"Selected Model: {selection4.primary_model.value}")
    print(f"Confidence: {selection4.confidence_score:.1%}")
    print(f"Reasoning: {selection4.reasoning}")


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent model selection for NFL predictions"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run example usage"
    )

    args = parser.parse_args()

    if args.example:
        example_usage()
    else:
        # Run with real data integration
        print("Integrate with auto_weekly_analyzer.py to get real data")


if __name__ == "__main__":
    main()
