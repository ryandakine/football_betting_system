#!/usr/bin/env python3
"""
Gold Standard Bridge with Standalone AI Council + Self-Learning Feedback System
Uses the standalone AI council without FastAPI dependencies and adds learning capabilities
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("gold_standard_bridge")

# Import the standalone AI council
try:
    from ai_council_standalone import HybridOptimizedGoldStandardMLBSystem

    LOGGER.info("✅ Successfully imported standalone AI Council")
    AI_COUNCIL_AVAILABLE = True
except ImportError as e:
    LOGGER.warning(f"⚠️ Could not import AI Council: {e}")
    LOGGER.warning("⚠️ Falling back to mock analysis")
    AI_COUNCIL_AVAILABLE = False

# Import the self-learning feedback system
try:
    from simple_learning_integration import (
        SimpleLearningTracker,
        add_learning_to_prediction,
        record_prediction_for_learning,
        update_outcome_for_learning,
    )

    LOGGER.info("🧠 Successfully imported Self-Learning Feedback System")
    LEARNING_SYSTEM_AVAILABLE = True
except ImportError as e:
    LOGGER.warning(f"⚠️ Could not import Learning System: {e}")
    LOGGER.warning("⚠️ Learning capabilities disabled")
    LEARNING_SYSTEM_AVAILABLE = False

# Import the enhanced learning system with travel/rest analysis
try:
    from enhanced_learning_system import EnhancedLearningSystem

    LOGGER.info(
        "✈️ Successfully imported Enhanced Learning System with Travel/Rest Analysis"
    )
    ENHANCED_LEARNING_AVAILABLE = True
except ImportError as e:
    LOGGER.warning(f"⚠️ Could not import Enhanced Learning System: {e}")
    LOGGER.warning("⚠️ Travel/rest analysis disabled")
    ENHANCED_LEARNING_AVAILABLE = False

# Initialize the AI council if available
if AI_COUNCIL_AVAILABLE:
    try:
        ai_system = HybridOptimizedGoldStandardMLBSystem(
            bankroll=500.0,
            base_unit_size=5.0,
            max_units=5,
            confidence_threshold=0.55,
            max_opportunities=50,
        )
        LOGGER.info("🤖 Standalone AI Council initialized successfully")
    except Exception as e:
        LOGGER.error(f"❌ Error initializing AI Council: {e}")
        AI_COUNCIL_AVAILABLE = False

# Initialize the learning system if available
if LEARNING_SYSTEM_AVAILABLE:
    try:
        learning_tracker = SimpleLearningTracker(
            db_path="data/gold_standard_learning.db"
        )
        LOGGER.info("🧠 Self-Learning Feedback System initialized successfully")
    except Exception as e:
        LOGGER.error(f"❌ Error initializing Learning System: {e}")
        LEARNING_SYSTEM_AVAILABLE = False

# Initialize the enhanced learning system if available
if ENHANCED_LEARNING_AVAILABLE:
    try:
        enhanced_learning_system = EnhancedLearningSystem(
            learning_db_path="data/enhanced_learning.db"
        )
        LOGGER.info(
            "✈️ Enhanced Learning System with Travel/Rest Analysis initialized successfully"
        )
    except Exception as e:
        LOGGER.error(f"❌ Error initializing Enhanced Learning System: {e}")
        ENHANCED_LEARNING_AVAILABLE = False


class GoldStandardBridgeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/ping":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            response = {
                "status": "ok",
                "message": "Gold Standard Bridge with AI Council + Learning + Travel/Rest Analysis is running",
                "ai_council_active": AI_COUNCIL_AVAILABLE,
                "learning_system_active": LEARNING_SYSTEM_AVAILABLE,
                "enhanced_learning_active": ENHANCED_LEARNING_AVAILABLE,
                "timestamp": datetime.now().isoformat(),
            }
            self.wfile.write(json.dumps(response).encode())

        elif parsed_path.path == "/learning-insights":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if LEARNING_SYSTEM_AVAILABLE:
                try:
                    insights = learning_tracker.get_insights()
                    response = {
                        "status": "success",
                        "learning_insights": insights,
                        "timestamp": datetime.now().isoformat(),
                    }
                except Exception as e:
                    LOGGER.error(f"❌ Error getting learning insights: {e}")
                    response = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
            else:
                response = {
                    "status": "error",
                    "error": "Learning system not available",
                    "timestamp": datetime.now().isoformat(),
                }

            self.wfile.write(json.dumps(response).encode())

        elif parsed_path.path == "/enhanced-insights":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if ENHANCED_LEARNING_AVAILABLE:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    insights = loop.run_until_complete(
                        enhanced_learning_system.get_comprehensive_insights()
                    )
                    loop.close()
                    response = {
                        "status": "success",
                        "enhanced_insights": insights,
                        "timestamp": datetime.now().isoformat(),
                    }
                except Exception as e:
                    LOGGER.error(f"❌ Error getting enhanced insights: {e}")
                    response = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
            else:
                response = {
                    "status": "error",
                    "error": "Enhanced learning system not available",
                    "timestamp": datetime.now().isoformat(),
                }

            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            data = []

        if parsed_path.path == "/opportunities":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if AI_COUNCIL_AVAILABLE:
                # Use the standalone AI council
                try:
                    # Run the AI council analysis asynchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Convert data to the format expected by the AI council
                    processed_data = []
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                # Add edge_cents calculation if not present
                                if "edge_cents" not in item:
                                    home_price = item.get("home_price", 0)
                                    away_price = item.get("away_price", 0)
                                    if home_price and away_price:
                                        # Simple edge calculation
                                        edge_cents = abs(home_price - away_price) * 0.1
                                        item["edge_cents"] = int(edge_cents)
                                    else:
                                        item["edge_cents"] = 10  # Default edge

                                processed_data.append(item)

                    # Run the AI council analysis
                    opportunities = loop.run_until_complete(
                        ai_system.analyze_opportunities_concurrently(processed_data)
                    )

                    # ENHANCEMENT: Apply learning to opportunities if learning system is available
                    enhanced_opportunities = []
                    if ENHANCED_LEARNING_AVAILABLE and opportunities:
                        LOGGER.info(
                            "🧠 Applying enhanced learning with travel/rest analysis..."
                        )

                        for opp in opportunities:
                            # Create base prediction for learning
                            base_prediction = {
                                "game_id": opp.get("game_id", "unknown"),
                                "home_team": opp.get("home_team", ""),
                                "away_team": opp.get("away_team", ""),
                                "predicted_winner": opp.get("team", ""),
                                "confidence": opp.get("confidence", 75)
                                / 100.0,  # Convert to 0-1 scale
                                "stake": opp.get("stake", 0.0),
                                "odds": opp.get("odds", 0.0),
                                "model_name": "ai_council_ensemble",
                                "features": {
                                    "edge_cents": opp.get("edge_cents", 0),
                                    "bet_type": opp.get("bet_type", "moneyline"),
                                    "confidence_original": opp.get("confidence", 75),
                                },
                            }

                            # Create game data for enhanced analysis
                            game_data = {
                                "game_id": opp.get("game_id", "unknown"),
                                "home_team": opp.get("home_team", ""),
                                "away_team": opp.get("away_team", ""),
                                "odds": opp.get("odds", 0.0),
                                "edge_cents": opp.get("edge_cents", 0),
                                "game_date": opp.get(
                                    "game_date", datetime.now().strftime("%Y-%m-%d")
                                ),
                                "home_pitcher": opp.get("home_pitcher"),
                                "away_pitcher": opp.get("away_pitcher"),
                                "away_team_city": opp.get("away_team_city", ""),
                                "home_team_city": opp.get("home_team_city", ""),
                                "travel_date": opp.get("travel_date", ""),
                                "travel_distance": opp.get("travel_distance", 0),
                            }

                            # Apply enhanced learning with travel/rest analysis
                            enhanced_prediction = loop.run_until_complete(
                                enhanced_learning_system.enhance_prediction_comprehensive(
                                    base_prediction, game_data
                                )
                            )

                            # Record for learning
                            prediction_id = loop.run_until_complete(
                                enhanced_learning_system.record_prediction_with_travel_rest(
                                    enhanced_prediction, game_data
                                )
                            )

                            # Create enhanced opportunity
                            enhanced_opp = opp.copy()
                            enhanced_opp["learning_prediction_id"] = prediction_id
                            enhanced_opp["original_confidence"] = opp.get(
                                "confidence", 75
                            )
                            enhanced_opp["enhanced_confidence"] = int(
                                enhanced_prediction["confidence"] * 100
                            )
                            enhanced_opp["total_enhancement"] = enhanced_prediction.get(
                                "total_enhancement", 0
                            )
                            enhanced_opp["confidence"] = enhanced_opp[
                                "enhanced_confidence"
                            ]  # Use enhanced confidence

                            # Add travel/rest analysis details
                            if "travel_rest_analysis" in enhanced_prediction:
                                analysis = enhanced_prediction["travel_rest_analysis"]
                                enhanced_opp["travel_rest_analysis"] = {
                                    "away_travel_fatigue": analysis.get(
                                        "away_travel_fatigue", 0
                                    ),
                                    "home_travel_fatigue": analysis.get(
                                        "home_travel_fatigue", 0
                                    ),
                                    "away_pitcher_rest": analysis.get(
                                        "away_pitcher_rest_factor", 0
                                    ),
                                    "home_pitcher_rest": analysis.get(
                                        "home_pitcher_rest_factor", 0
                                    ),
                                    "overall_impact": analysis.get(
                                        "overall_travel_rest_impact", 0
                                    ),
                                    "recommendation": analysis.get(
                                        "recommendation", "N/A"
                                    ),
                                }

                            enhanced_opportunities.append(enhanced_opp)

                            LOGGER.info(
                                f"🎯 Enhanced opportunity: {opp.get('team', 'Unknown')} "
                                f"confidence {opp.get('confidence', 75)} → {enhanced_opp['enhanced_confidence']} "
                                f"(total enhancement: {enhanced_opp['total_enhancement']:+.3f})"
                            )

                        # Use enhanced opportunities
                        opportunities = enhanced_opportunities
                        LOGGER.info(
                            f"🧠 Enhanced learning applied to {len(enhanced_opportunities)} opportunities"
                        )

                    elif LEARNING_SYSTEM_AVAILABLE and opportunities:
                        LOGGER.info(
                            "🧠 Applying basic learning enhancement to opportunities..."
                        )

                        for opp in opportunities:
                            # Create base prediction for learning
                            base_prediction = {
                                "game_id": opp.get("game_id", "unknown"),
                                "home_team": opp.get("home_team", ""),
                                "away_team": opp.get("away_team", ""),
                                "predicted_winner": opp.get("team", ""),
                                "confidence": opp.get("confidence", 75)
                                / 100.0,  # Convert to 0-1 scale
                                "stake": opp.get("stake", 0.0),
                                "odds": opp.get("odds", 0.0),
                                "model_name": "ai_council_ensemble",
                                "features": {
                                    "edge_cents": opp.get("edge_cents", 0),
                                    "bet_type": opp.get("bet_type", "moneyline"),
                                    "confidence_original": opp.get("confidence", 75),
                                },
                            }

                            # Create game data for learning
                            game_data = {
                                "game_id": opp.get("game_id", "unknown"),
                                "home_team": opp.get("home_team", ""),
                                "away_team": opp.get("away_team", ""),
                                "odds": opp.get("odds", 0.0),
                                "edge_cents": opp.get("edge_cents", 0),
                            }

                            # Apply learning enhancement
                            enhanced_prediction = add_learning_to_prediction(
                                base_prediction, game_data, learning_tracker
                            )

                            # Record for learning
                            prediction_id = record_prediction_for_learning(
                                enhanced_prediction, learning_tracker
                            )

                            # Create enhanced opportunity
                            enhanced_opp = opp.copy()
                            enhanced_opp["learning_prediction_id"] = prediction_id
                            enhanced_opp["original_confidence"] = opp.get(
                                "confidence", 75
                            )
                            enhanced_opp["enhanced_confidence"] = int(
                                enhanced_prediction["confidence"] * 100
                            )
                            enhanced_opp["learning_boost"] = enhanced_prediction.get(
                                "learning_boost", 0
                            )
                            enhanced_opp["confidence"] = enhanced_opp[
                                "enhanced_confidence"
                            ]  # Use enhanced confidence

                            enhanced_opportunities.append(enhanced_opp)

                            LOGGER.info(
                                f"🎯 Enhanced opportunity: {opp.get('team', 'Unknown')} "
                                f"confidence {opp.get('confidence', 75)} → {enhanced_opp['enhanced_confidence']} "
                                f"(boost: {enhanced_opp['learning_boost']:+.3f})"
                            )

                        # Use enhanced opportunities
                        opportunities = enhanced_opportunities
                        LOGGER.info(
                            f"🧠 Learning enhancement applied to {len(enhanced_opportunities)} opportunities"
                        )

                    # Get performance stats
                    stats = ai_system.get_performance_stats()

                    # Save recommendations if any found
                    if opportunities:
                        filename = loop.run_until_complete(
                            ai_system.save_recommendations_async(opportunities)
                        )
                        LOGGER.info(
                            f"💾 AI Council saved {len(opportunities)} opportunities to {filename}"
                        )
                    else:
                        filename = "no_opportunities_found"
                        LOGGER.info("⚠️ AI Council found no opportunities")

                    response = {
                        "status": "analysis_complete",
                        "ai_council_active": True,
                        "learning_system_active": LEARNING_SYSTEM_AVAILABLE,
                        "confidence_score": 0.85,  # AI council confidence
                        "predictions": [
                            {
                                "team": opp.get("team", "Unknown"),
                                "bet_type": opp.get("bet_type", "moneyline"),
                                "confidence": opp.get("confidence", 75),
                                "original_confidence": opp.get(
                                    "original_confidence", 75
                                ),
                                "learning_boost": opp.get("learning_boost", 0),
                                "learning_prediction_id": opp.get(
                                    "learning_prediction_id", ""
                                ),
                            }
                            for opp in opportunities[:3]  # Top 3 predictions
                        ],
                        "recommended_bets": [
                            {
                                "description": f"{opp.get('team', 'Team')} {opp.get('bet_type', 'ML')}",
                                "edge": opp.get("edge_cents", 5.0)
                                / 100.0,  # Convert cents to decimal
                                "confidence": int(opp.get("confidence", 75)),
                                "learning_enhanced": "learning_prediction_id" in opp,
                            }
                            for opp in opportunities[:2]  # Top 2 bets
                        ],
                        "analysis_time": datetime.now().isoformat(),
                        "games_analyzed": len(processed_data),
                        "performance_stats": stats,
                        "file": filename,
                    }

                    loop.close()

                except Exception as e:
                    LOGGER.error(f"❌ Error in AI Council analysis: {e}")
                    response = self._get_mock_response(data)

            else:
                # Fall back to mock analysis
                response = self._get_mock_response(data)

            LOGGER.info(
                f"🤖 Analysis complete: {len(response.get('predictions', []))} predictions"
            )
            self.wfile.write(json.dumps(response).encode())

        elif parsed_path.path == "/update-outcomes":
            """New endpoint to update prediction outcomes for learning"""
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if LEARNING_SYSTEM_AVAILABLE:
                try:
                    outcomes = data if isinstance(data, list) else []
                    processed_count = 0

                    for outcome in outcomes:
                        prediction_id = outcome.get("learning_prediction_id")
                        actual_winner = outcome.get("actual_winner")
                        profit = outcome.get("profit", 0.0)

                        if prediction_id and actual_winner:
                            update_outcome_for_learning(
                                prediction_id, actual_winner, profit, learning_tracker
                            )
                            processed_count += 1

                    # Run learning analysis after processing outcomes
                    learning_tracker.analyze_and_learn()

                    # Get updated insights
                    insights = learning_tracker.get_insights()

                    response = {
                        "status": "success",
                        "outcomes_processed": processed_count,
                        "learning_analysis_complete": True,
                        "learning_insights": insights,
                        "timestamp": datetime.now().isoformat(),
                    }

                    LOGGER.info(f"✅ Processed {processed_count} outcomes for learning")

                except Exception as e:
                    LOGGER.error(f"❌ Error processing outcomes: {e}")
                    response = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
            else:
                response = {
                    "status": "error",
                    "error": "Learning system not available",
                    "timestamp": datetime.now().isoformat(),
                }

            self.wfile.write(json.dumps(response).encode())

        elif parsed_path.path == "/sentiment":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Mock sentiment analysis response
            response = {
                "status": "sentiment_analysis_complete",
                "sentiment_score": 0.65,
                "confidence": 0.78,
                "analysis_time": datetime.now().isoformat(),
            }

            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def _get_mock_response(self, data):
        """Generate mock response when AI Council is not available"""
        return {
            "status": "mock_analysis_complete",
            "ai_council_active": False,
            "learning_system_active": LEARNING_SYSTEM_AVAILABLE,
            "confidence_score": 0.60,
            "predictions": [
                {
                    "team": "Mock Team",
                    "bet_type": "moneyline",
                    "confidence": 65,
                    "original_confidence": 65,
                    "learning_boost": 0,
                    "learning_prediction_id": "",
                }
            ],
            "recommended_bets": [
                {
                    "description": "Mock Team ML",
                    "edge": 0.05,
                    "confidence": 65,
                    "learning_enhanced": False,
                }
            ],
            "analysis_time": datetime.now().isoformat(),
            "games_analyzed": len(data) if isinstance(data, list) else 0,
            "performance_stats": {"mock": True},
            "file": "mock_analysis.json",
        }

    def log_message(self, format, *args):
        """Override to use our logger"""
        LOGGER.info(format % args)


def run_server(port=8767):
    """Run the HTTP server"""
    server_address = ("", port)
    httpd = HTTPServer(server_address, GoldStandardBridgeHandler)

    LOGGER.info(f"🚀 Starting Gold Standard Bridge with Learning on port {port}")
    LOGGER.info(
        f"🤖 AI Council Status: {'✅ Active' if AI_COUNCIL_AVAILABLE else '❌ Inactive'}"
    )
    LOGGER.info(
        f"🧠 Learning System Status: {'✅ Active' if LEARNING_SYSTEM_AVAILABLE else '❌ Inactive'}"
    )
    LOGGER.info("📡 Available endpoints:")
    LOGGER.info("   GET  /ping")
    LOGGER.info("   POST /opportunities")
    LOGGER.info("   POST /sentiment")
    LOGGER.info("   GET  /learning-insights")
    LOGGER.info("   GET  /enhanced-insights")
    LOGGER.info("   POST /update-outcomes")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("⚠️ Server stopped by user")
    finally:
        httpd.server_close()
        LOGGER.info("🔚 Server closed")


if __name__ == "__main__":
    run_server()
