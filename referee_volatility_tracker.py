#!/usr/bin/env python3
import requests, re, json, sqlite3, os
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import pandas as pd

def scrape_officials(): 
    resp = requests.get("https://www.footballzebras.com/category/referee-assignments/"); soup = BeautifulSoup(resp.content, 'html.parser'); pdf_links = [a['href'] for a in soup.find_all('a') if 'pdf' in a.get('href', '').lower()]; return pdf_links[0] if pdf_links else None

def parse_crew_data(pdf_url):
    resp = requests.get(pdf_url); open('temp_roster.pdf', 'wb').write(resp.content); crew_data = {}; text = os.popen('pdftotext temp_roster.pdf -').read(); lines = [l.strip() for l in text.split('\n') if l.strip()]; current_crew = None
    for line in lines:
        if 'Referee:' in line: current_crew = re.search(r'Referee:\s*(.+)', line).group(1).strip(); crew_data[current_crew] = {'referee': current_crew, 'umpire': '', 'down_judge': ''}
        elif 'Umpire:' in line and current_crew: crew_data[current_crew]['umpire'] = re.search(r'Umpire:\s*(.+)', line).group(1).strip()
        elif 'Down Judge:' in line and current_crew: crew_data[current_crew]['down_judge'] = re.search(r'Down Judge:\s*(.+)', line).group(1).strip()
    return crew_data

def track_ref_changes():
    conn = sqlite3.connect('referee_tracking.db'); conn.execute('CREATE TABLE IF NOT EXISTS ref_history (week INT, crew_chief TEXT, referee TEXT, umpire TEXT, down_judge TEXT, timestamp TEXT)'); pdf_url = scrape_officials(); crew_data = parse_crew_data(pdf_url) if pdf_url else {}; current_week = datetime.now().isocalendar()[1]
    for crew_chief, positions in crew_data.items(): conn.execute('INSERT INTO ref_history VALUES (?, ?, ?, ?, ?, ?)', (current_week, crew_chief, positions['referee'], positions['umpire'], positions['down_judge'], datetime.now().isoformat()))
    conn.commit(); changes = conn.execute('SELECT * FROM ref_history WHERE week >= ? ORDER BY week DESC', (current_week-5,)).fetchall(); return analyze_changes(changes, current_week)

def analyze_changes(history, current_week):
    changes = []; crew_tracking = {}
    for week, crew_chief, referee, umpire, down_judge, timestamp in history:
        if crew_chief not in crew_tracking: crew_tracking[crew_chief] = {}
        if week not in crew_tracking[crew_chief]: crew_tracking[crew_chief][week] = {'referee': referee, 'umpire': umpire, 'down_judge': down_judge}
    for crew_chief, weeks in crew_tracking.items():
        week_list = sorted(weeks.keys())
        for i in range(1, len(week_list)): 
            curr, prev = weeks[week_list[i]], weeks[week_list[i-1]]
            if current_week - week_list[i-1] > 3 and (curr['referee'] != prev['referee'] or curr['umpire'] != prev['umpire'] or curr['down_judge'] != prev['down_judge']): changes.append({'crew': crew_chief, 'week': week_list[i], 'change': f"{prev} -> {curr}"})
    return changes

def trigger_alert(changes):
    for change in changes: print(f"🚨 HIGH VOLATILITY: {change['crew']} Week {change['week']} - ref swap: {change['change']}, check injury reports + betting volume"); os.system(f"curl -X POST https://api.pushover.net/1/messages.json -d 'token=YOUR_TOKEN&user=YOUR_USER&message=ref swap: {change['change']}, check injury reports + betting volume'")

def backtest_thursday_nights():
    thursday_games = ['KC_vs_BAL_TNF', 'BUF_vs_MIA_TNF', 'SF_vs_SEA_TNF', 'DAL_vs_NYG_TNF', 'NE_vs_NYJ_TNF']; results = []
    for game in thursday_games: volatility = len([c for c in track_ref_changes() if 'TNF' in game]) > 0; results.append({'game': game, 'ref_volatility': volatility, 'outcome': 'higher_variance' if volatility else 'normal'})
    return results

if __name__ == "__main__": changes = track_ref_changes(); trigger_alert(changes); backtest_results = backtest_thursday_nights(); print(f"✅ Referee tracking complete: {len(changes)} volatility alerts, backtest: {backtest_results}")
