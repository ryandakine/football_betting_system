#!/usr/bin/env python3
"""
Quick script to get Monday Night Football referee assignment
"""
import requests
from bs4 import BeautifulSoup
import os

def get_mnf_referee():
    """Scrape Football Zebras for Monday Night Football referee."""

    print("🏈 Fetching Monday Night Football referee assignment...")

    # Football Zebras URL for current week
    url = "https://www.footballzebras.com/category/assignments/"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for Monday Night Football
        text = soup.get_text()

        # Find PHI @ GB or Eagles @ Packers
        if "Eagles" in text or "Packers" in text or "PHI" in text:
            print("\n✅ FOUND MONDAY NIGHT FOOTBALL!")

            # Extract referee name (usually after "Referee:" or in specific format)
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if 'Monday' in line or 'PHI' in line or 'GB' in line or 'Eagles' in line or 'Packers' in line:
                    # Print surrounding context
                    start = max(0, i-3)
                    end = min(len(lines), i+10)
                    print('\n'.join(lines[start:end]))
                    break
        else:
            print("❌ Eagles @ Packers not found in current assignments")
            print("\nShowing latest Monday Night assignments:")
            # Show whatever Monday games are listed
            for i, line in enumerate(text.split('\n')):
                if 'Monday' in line:
                    print(lines[max(0,i-2):min(len(lines),i+5)])

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTry manually checking: https://www.footballzebras.com/category/assignments/")

if __name__ == "__main__":
    get_mnf_referee()
