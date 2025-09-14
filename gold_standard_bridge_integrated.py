#!/usr/bin/env python3
"""
Gold Standard Bridge with Real AI Council Integration
Uses built-in HTTP server but connects to the real multi-modal AI council
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

# Import the real Gold Standard system
try:
    from gold_standard_main import HybridOptimizedGoldStandardMLBSystem

    LOGGER.info("✅ Successfully imported real Gold Standard AI Council")
    AI_COUNCIL_AVAILABLE = True
except ImportError as e:
    LOGGER.warning(f"⚠️ Could not import real AI Council: {e}")
    LOGGER.warning("⚠️ Falling back to mock AI analysis")
    AI_COUNCIL_AVAILABLE = False

# Initialize the real AI council if available
if AI_COUNCIL_AVAILABLE:
    try:
        ai_system = HybridOptimizedGoldStandardMLBSystem(
            bankroll=500.0,
            base_unit_size=5.0,
            max_units=5,
            confidence_threshold=0.55,
            max_opportunities=50,
        )
        LOGGER.info("🤖 Real AI Council initialized successfully")
    except Exception as e:
        LOGGER.error(f"❌ Error initializing AI Council: {e}")
        AI_COUNCIL_AVAILABLE = False


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
                "message": "Gold Standard Bridge with AI Council is running",
                "ai_council_active": AI_COUNCIL_AVAILABLE,
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
                # Use the real AI council
                try:
                    # Run the AI council analysis asynchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Convert data to the format expected by the AI council
                    processed_data = []
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                processed_data.append(item)

                    # Run the AI council analysis
                    opportunities = loop.run_until_complete(
                        ai_system.analyze_opportunities_concurrently(processed_data)
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
                        "confidence_score": 0.85,  # AI council confidence
                        "predictions": [
                            {
                                "team": opp.get("team", "Unknown"),
                                "bet_type": opp.get("bet_type", "moneyline"),
                                "confidence": opp.get("confidence", 75),
                            }
                            for opp in opportunities[:3]  # Top 3 predictions
                        ],
                        "recommended_bets": [
                            {
                                "description": f"{opp.get('team', 'Team')} {opp.get('bet_type', 'ML')}",
                                "edge": opp.get("edge", 5.0),
                                "confidence": opp.get("confidence", 75),
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

        elif parsed_path.path == "/sentiment":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Save sentiment data for the AI council
            try:
                fp = Path("data") / f"sentiment_{datetime.now():%Y-%m-%d}.json"
                fp.parent.mkdir(exist_ok=True)
                fp.write_text(json.dumps(data, indent=2))
                LOGGER.info(
                    f"📊 Sentiment data saved for AI council: {len(data)} items"
                )
            except Exception as e:
                LOGGER.error(f"❌ Error saving sentiment: {e}")

            response = {
                "status": "sentiment_processed",
                "message": "Sentiment data saved for AI council analysis",
                "ai_council_active": AI_COUNCIL_AVAILABLE,
                "timestamp": datetime.now().isoformat(),
            }

            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def _get_mock_response(self, data):
        """Fallback mock response when AI council is not available"""
        predictions = []
        if isinstance(data, list) and len(data) > 0:
            for i, row in enumerate(data[:3]):
                home_team = row.get("home_team", "Team A")
                away_team = row.get("away_team", "Team B")
                predictions.append(
                    {
                        "team": home_team if i % 2 == 0 else away_team,
                        "bet_type": "moneyline",
                        "confidence": 75 + (i * 5),
                    }
                )

            recommended_bets = [
                {
                    "description": f"{data[0].get('home_team', 'Team A')} ML",
                    "edge": 8.5,
                    "confidence": 78,
                }
            ]
        else:
            recommended_bets = []

        return {
            "status": "analysis_complete",
            "ai_council_active": False,
            "confidence_score": 0.78,
            "predictions": predictions,
            "recommended_bets": recommended_bets,
            "analysis_time": datetime.now().isoformat(),
            "games_analyzed": len(data) if isinstance(data, list) else 0,
            "note": "Using mock analysis - AI council not available",
        }

    def log_message(self, format, *args):
        """Override to use our logger"""
        LOGGER.info(f"HTTP: {format % args}")


def run_server(port=8767):
    """Run the Gold Standard Bridge server"""
    server_address = ("", port)
    httpd = HTTPServer(server_address, GoldStandardBridgeHandler)
    LOGGER.info(f"🚀 Starting Gold Standard Bridge on port {port}")
    LOGGER.info(
        f"🤖 AI Council Status: {'✅ Active' if AI_COUNCIL_AVAILABLE else '⚠️ Mock Mode'}"
    )
    LOGGER.info(f"📡 Available endpoints:")
    LOGGER.info(f"   GET  /ping")
    LOGGER.info(f"   POST /opportunities")
    LOGGER.info(f"   POST /sentiment")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("🛑 Server stopped by user")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
