#!/usr/bin/env python3
"""
Simple HTTP Server to mock Gold Standard Bridge
"""

import json
import logging
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("simple_bridge")


class SimpleBridgeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/ping":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            response = {
                "status": "ok",
                "message": "Simple Gold Standard Bridge is running",
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

            # Mock AI analysis
            predictions = []
            if isinstance(data, list) and len(data) > 0:
                for i, row in enumerate(data[:3]):  # Limit to 3 predictions
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

            response = {
                "status": "analysis_complete",
                "confidence_score": 0.78,
                "predictions": predictions,
                "recommended_bets": recommended_bets,
                "analysis_time": datetime.now().isoformat(),
                "games_analyzed": len(data) if isinstance(data, list) else 0,
            }

            LOGGER.info(f"🤖 Mock AI analysis complete: {len(predictions)} predictions")
            self.wfile.write(json.dumps(response).encode())

        elif parsed_path.path == "/sentiment":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            response = {
                "status": "sentiment_processed",
                "message": "Mock sentiment analysis complete",
                "timestamp": datetime.now().isoformat(),
            }

            LOGGER.info(f"📊 Mock sentiment analysis complete")
            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def log_message(self, format, *args):
        """Override to use our logger"""
        LOGGER.info(f"HTTP: {format % args}")


def run_server(port=8767):
    """Run the simple bridge server"""
    server_address = ("", port)
    httpd = HTTPServer(server_address, SimpleBridgeHandler)
    LOGGER.info(f"🚀 Starting Simple Gold Standard Bridge on port {port}")
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
