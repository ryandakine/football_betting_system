import json
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

fresh_odds = {}


class OddsHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/update-odds":
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)

                # Try JSON first
                try:
                    game_data = json.loads(post_data.decode("utf-8"))
                except:
                    # If JSON fails, try form data
                    parsed_data = urllib.parse.parse_qs(post_data.decode("utf-8"))
                    game_data = json.loads(parsed_data["data"][0])

                # Store and log as before
                fresh_odds[game_data["gameId"]] = game_data
                print(f"📊 Updated odds for: {game_data['matchup']}")

                if game_data.get("isTigersGame"):
                    print(f"🐅 FRESH TIGERS ODDS RECEIVED!")
                    print(
                        f"   Fresh Tigers odds: {game_data['bestOdds']['away']['price']}"
                    )

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "success"}')

            except Exception as e:
                print(f"❌ Error: {e}")
                self.send_response(400)
                self.end_headers()


if __name__ == "__main__":
    print("🚀 Starting odds receiver server...")
    server = HTTPServer(("localhost", 5000), OddsHandler)
    server.serve_forever()
