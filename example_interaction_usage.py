#!/usr/bin/env python3
"""
Example: How to use NFL Interaction World Model in your betting workflow
"""

from nfl_interaction_world_model_v2 import NFLInteractionWorldModel

# Initialize model (loads from cache if available)
interaction_model = NFLInteractionWorldModel()

# Example 1: Record a prediction for learning
# ==========================================
game_id = "Chiefs @ Bills"

# Get predictions from your 12 models (0-1 confidence scale)
model_predictions = {
    'referee': 0.68,        # Referee model predicts 68% confidence
    'deepseek': 0.72,       # DeepSeek predicts 72%
    'contrarian': 0.45,     # Contrarian sees trap (inverse)
    'handle': 0.62,         # Handle detector 62%
    'weather': 0.71,        # Weather favors UNDER 71%
    'line_shopping': 0.65,  # Line shopping found value
    'kelly': 0.60,          # Kelly sizing suggests bet
    'trend': 0.58,          # Trend analysis 58%
    'injury': 0.52,         # Injury impact minimal
    'rest': 0.55,           # Rest days neutral
    'divisional': 0.64,     # Divisional matchup 64%
    'steam': 0.69           # Steam move detected 69%
}

# Record for learning (result can be None before game settles)
interaction_model.record_prediction_batch(
    game_id=game_id,
    model_predictions=model_predictions,
    actual_result=None  # Set to "WIN" or "LOSS" after game
)

# Example 2: Boost a prediction using learned interactions
# =========================================================
base_confidence = 0.65  # Your base prediction (65%)

# Apply interaction boost
boosted_confidence, boost_details = interaction_model.boost_prediction(
    base_confidence=base_confidence,
    model_predictions=model_predictions
)

print(f"\n🎯 PREDICTION BOOSTING:")
print(f"   Base Confidence: {base_confidence*100:.1f}%")
print(f"   Boosted Confidence: {boosted_confidence*100:.1f}%")
print(f"   Total Boost: +{boost_details['total_boost']*100:.1f}%")
print(f"   Interactions Fired: {boost_details['interaction_count']}")

if boost_details['boosts_applied']:
    print(f"\n   Active Interactions:")
    for boost in boost_details['boosts_applied']:
        models = ' + '.join(boost['models'])
        print(f"      • {boost['type']}: {models} (+{boost['boost']*100:.1f}%)")

# Example 3: Check which interactions are currently active
# ========================================================
active = interaction_model.get_active_interactions(model_predictions)

print(f"\n🔥 ACTIVE INTERACTIONS:")
print(f"   2-way: {len(active['2way'])}")
print(f"   3-way: {len(active['3way'])}")

# Example 4: After game settles, record result
# ============================================
# After you know if bet won/lost:
# interaction_model.record_prediction_batch(
#     game_id=game_id,
#     model_predictions=model_predictions,
#     actual_result="WIN"  # or "LOSS"
# )

# Example 5: View model status
# ===========================
status = interaction_model.to_dict()
print(f"\n📊 MODEL STATUS:")
print(f"   2-way Interactions Learned: {status['interactions_2way_count']}")
print(f"   3-way Interactions Learned: {status['interactions_3way_count']}")
print(f"   Historical Records: {status['history_size']}")

print("\n✅ Integration example complete!")
print("   - Record predictions before each game")
print("   - Use boost_prediction() to enhance confidence")
print("   - Record results after games settle")
print("   - Model learns and improves over time")
