#!/usr/bin/env python3
"""
🏆 ROBUST GOLD STANDARD FILE SERVER
Handles N8N Gold Standard Bridge data with debug output
"""

import json
import os
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse


class GoldStandardHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        """Handle GET requests (health checks)"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Check if files exist
            files_status = {}
            try:
                files_to_check = [
                    ("main_feed", "data/latest_n8n_feed.json"),
                    ("status_file", "status/n8n_status.json"),
                    ("heartbeat", "status/last_heartbeat.json"),
                ]

                for file_type, file_path in files_to_check:
                    if os.path.exists(file_path):
                        stat = os.stat(file_path)
                        files_status[file_type] = {
                            "exists": True,
                            "last_modified": datetime.fromtimestamp(
                                stat.st_mtime
                            ).isoformat(),
                            "size_bytes": stat.st_size,
                        }
                    else:
                        files_status[file_type] = {
                            "exists": False,
                            "last_modified": None,
                        }
            except Exception as e:
                files_status["error"] = str(e)

            health_data = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "server_mode": "robust_gold_standard_processor",
                "uptime_seconds": time.time() - server_start_time,
                "files": files_status,
            }

            self.wfile.write(json.dumps(health_data, indent=2).encode())
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 💓 Health check requested")

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def do_POST(self):
        """Handle POST requests (data from N8N)"""
        try:
            if self.path == "/write":
                # Read the posted data with size limit
                content_length = int(self.headers.get("Content-Length", 0))

                # Safety check for extremely large payloads
                if content_length > 50 * 1024 * 1024:  # 50MB limit
                    self.send_response(413)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    error_response = {"error": "Payload too large", "max_size": "50MB"}
                    self.wfile.write(json.dumps(error_response).encode())
                    return

                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode("utf-8"))

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] 📦 COMPLETE PACKAGE RECEIVED"
                )

                # DEBUG: Print what we actually received
                print(f"           🔍 DEBUG: Received data keys: {list(data.keys())}")
                print(f"           🔍 DEBUG: Data type: {type(data)}")

                # Process the data (handles multiple formats)
                files_written = self.process_data_intelligently(data)

                # Send success response
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                response = {
                    "status": "success",
                    "files_written": files_written,
                    "timestamp": datetime.now().isoformat(),
                    "data_format_detected": self.detect_data_format(data),
                }

                self.wfile.write(json.dumps(response, indent=2).encode())

                print(f"           ✅ PROCESSING COMPLETE")
                print(f"           📁 Files written: {', '.join(files_written)}")

            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Endpoint not found")

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ ERROR: {str(e)}")
            self.send_response(500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            error_response = {"error": str(e), "timestamp": datetime.now().isoformat()}
            self.wfile.write(json.dumps(error_response, indent=2).encode())

    def detect_data_format(self, data):
        """Detect what format the data is in"""
        if isinstance(data, dict):
            if (
                "export_type" in data
                and data["export_type"] == "complete_gold_standard_package"
            ):
                return "complete_package_format"
            elif "main_feed" in data and "status" in data and "health" in data:
                return "separated_components_format"
            elif "live_opportunities" in data and "avoid_recommendations" in data:
                return "direct_feed_format"
            else:
                return f"unknown_dict_format_keys_{list(data.keys())}"
        else:
            return f"unexpected_type_{type(data)}"

    def process_data_intelligently(self, data):
        """Process data regardless of format"""
        files_written = []

        try:
            # Ensure directories exist
            os.makedirs("data", exist_ok=True)
            os.makedirs("status", exist_ok=True)

            format_type = self.detect_data_format(data)
            print(f"           🎯 Detected format: {format_type}")

            if format_type == "complete_package_format":
                # Handle: {export_type: '...', main_feed: {...}, status: {...}, health: {...}}
                files_written = self.process_complete_package(data)

            elif format_type == "separated_components_format":
                # Handle: {main_feed: {...}, status: {...}, health: {...}}
                files_written = self.process_separated_components(data)

            elif format_type == "direct_feed_format":
                # Handle: Direct feed data {live_opportunities: [...], ...}
                files_written = self.process_direct_feed(data)

            else:
                # Unknown format - save as debug file
                debug_path = "data/debug_unknown_format.json"
                with open(debug_path, "w") as f:
                    json.dump(data, f, indent=2)
                files_written.append("debug_unknown_format")
                print(f"           ⚠️  Unknown format saved as debug file")

        except Exception as e:
            print(f"           ❌ Processing error: {str(e)}")
            # Save error data for debugging
            error_path = "data/error_debug_data.json"
            with open(error_path, "w") as f:
                json.dump({"error": str(e), "data": data}, f, indent=2)
            files_written.append("error_debug")
            raise e

        return files_written

    def process_complete_package(self, data):
        """Process complete package format"""
        files_written = []

        if "main_feed" in data:
            main_feed_path = "data/latest_n8n_feed.json"
            with open(main_feed_path, "w") as f:
                json.dump(data["main_feed"], f, indent=2)
            files_written.append("main_feed")

            opportunities = len(data["main_feed"].get("live_opportunities", []))
            avoid_count = len(data["main_feed"].get("avoid_recommendations", []))
            print(
                f"           📊 Main feed: {opportunities} opportunities, {avoid_count} avoid"
            )

        if "status" in data:
            status_path = "status/n8n_status.json"
            with open(status_path, "w") as f:
                json.dump(data["status"], f, indent=2)
            files_written.append("status")
            print(
                f"           ⚡ Status: {data['status'].get('system_status', 'unknown')}"
            )

        if "health" in data:
            health_path = "status/last_heartbeat.json"
            with open(health_path, "w") as f:
                json.dump(data["health"], f, indent=2)
            files_written.append("health")
            print(f"           💓 Heartbeat: {data['health'].get('status', 'unknown')}")

        return files_written

    def process_separated_components(self, data):
        """Process separated components format"""
        return self.process_complete_package(data)  # Same logic

    def process_direct_feed(self, data):
        """Process direct feed format"""
        files_written = []

        # Create a properly structured main feed
        structured_feed = {
            "export_metadata": {
                "source": "n8n_direct_feed",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0",
                "integration_type": "gold_standard_feed",
            },
            "live_opportunities": data.get("live_opportunities", []),
            "avoid_recommendations": data.get("avoid_recommendations", []),
            "market_snapshot": data.get("market_snapshot", {}),
            "current_performance": data.get("current_performance", {}),
        }

        main_feed_path = "data/latest_n8n_feed.json"
        with open(main_feed_path, "w") as f:
            json.dump(structured_feed, f, indent=2)
        files_written.append("main_feed")

        opportunities = len(structured_feed["live_opportunities"])
        avoid_count = len(structured_feed["avoid_recommendations"])
        print(
            f"           📊 Direct feed: {opportunities} opportunities, {avoid_count} avoid"
        )

        # Create basic status file
        status_data = {
            "system_status": "operational",
            "last_run": datetime.now().isoformat(),
            "opportunities_found": opportunities,
            "avoid_recommendations": avoid_count,
            "source": "direct_feed_processing",
        }

        status_path = "status/n8n_status.json"
        with open(status_path, "w") as f:
            json.dump(status_data, f, indent=2)
        files_written.append("status")

        # Create basic health file
        health_data = {
            "heartbeat": datetime.now().isoformat(),
            "status": "alive",
            "source": "direct_feed_processor",
            "opportunities_count": opportunities,
        }

        health_path = "status/last_heartbeat.json"
        with open(health_path, "w") as f:
            json.dump(health_data, f, indent=2)
        files_written.append("health")

        return files_written

    def log_message(self, format, *args):
        """Suppress default HTTP logging"""
        pass


def run_server():
    """Run the Gold Standard file server"""
    global server_start_time
    server_start_time = time.time()

    print("🚀 ROBUST GOLD STANDARD FILE SERVER STARTING")
    print(f"   📡 Listening on: http://localhost:8765")
    print(f"   📁 Writing files to: {os.path.abspath('.')}/")
    print(f"   🧠 Mode: Intelligent data processing")
    print(f"   🔍 Handles: Complete packages, separated components, direct feeds")
    print(f"   ❤️  Health Check: http://localhost:8765/health")
    print(f"   ⏹️  Press Ctrl+C to stop")
    print("=" * 70)

    try:
        server = HTTPServer(("localhost", 8765), GoldStandardHandler)
        server.serve_forever()

    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🛑 Server stopped by user")
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ Server error: {str(e)}")


if __name__ == "__main__":
    run_server()
