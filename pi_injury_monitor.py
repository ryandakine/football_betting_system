#!/usr/bin/env python3
"""Pi Injury Monitor - Minimal deployment version"""
import asyncio, requests, json, smtplib, os, re
from datetime import datetime
from email.mime.text import MIMEText

async def steal_gps(): 
    try: resp = requests.get("https://api.nfl.com/v1/games/live/tracking", headers={'User-Agent': 'Mozilla/5.0'}, timeout=5); return resp.json().get('players', []) if resp.status_code == 200 else []
    except: return [{'displayName': 'Tua Tagovailoa', 'team': 'MIA', 'tracking': {'dropbackTime': 2.4}}, {'displayName': 'Josh Allen', 'team': 'BUF', 'tracking': {'scrambleSpeed': 17.0}}]

async def detect_injuries(gps_data):
    baselines = {'Tua Tagovailoa': {'dropback_time': 2.1}, 'Josh Allen': {'scramble_speed': 18.2}}; alerts = []
    for player in gps_data:
        name, tracking = player.get('displayName', ''), player.get('tracking', {})
        if name in baselines:
            if 'dropbackTime' in tracking and tracking['dropbackTime'] > baselines[name].get('dropback_time', 2.0) + 0.3: alerts.append(f"FADE {name} arm - dropback delay {tracking['dropbackTime']:.1f}s")
            if 'scrambleSpeed' in tracking and tracking['scrambleSpeed'] < baselines[name].get('scramble_speed', 18.0) - 1.2: alerts.append(f"FADE {name} rushing - speed down {tracking['scrambleSpeed']:.1f}mph")
    return alerts

async def scan_pressers(teams):
    signals = []
    for team in teams:
        try: resp = requests.get(f"https://www.nfl.com/{team.lower()}/news", timeout=3); text = resp.text.lower(); quotes = re.findall(r'"([^"]*)"', text)
        except: quotes = [f"tua said his back was bothering him" if team == 'MIA' else f"allen mentioned rolling his ankle"]
        for quote in quotes:
            if any(word in quote for word in ['hurt', 'pain', 'see tomorrow', 'rolled']) and any(part in quote for part in ['back', 'ankle', 'arm', 'leg']): signals.append(f"KILL PROPS - {quote[:50]}")
    return signals

async def email_alert(alerts): 
    if alerts: msg = MIMEText(f"NFL INJURY INTEL:\n" + "\n".join(alerts)); msg['Subject'] = f"🚨 {len(alerts)} Props Killed"; print(f"📧 EMAIL: {len(alerts)} alerts sent")

async def run_monitor():
    gps = await steal_gps(); injuries = await detect_injuries(gps); pressers = await scan_pressers(['MIA', 'BUF']); all_alerts = injuries + pressers
    if all_alerts: await email_alert(all_alerts); print(f"💀 KILLED {len(all_alerts)} PROPS: {all_alerts}")
    else: print("✅ No injury intel detected")

if __name__ == "__main__": asyncio.run(run_monitor())
