#!/usr/bin/env python3
"""
Analyze Completed Games
========================
Retrospective analysis using enhanced intelligence system on games that already happened.
Shows what the system would have recommended vs actual results.
"""

import asyncio
import json
from pathlib import Path
from datetime import date

from simple_narrative_scraper import SimpleNarrativeScraper

DATA_DIR = Path("data/referee_conspiracy")


async def analyze_completed_game(
    game_id: str,
    home_team: str,
    away_team: str,
    actual_result: dict,
    scraper: SimpleNarrativeScraper,
):
    """Analyze a completed game"""
    
    print(f"\n{'='*80}")
    print(f"🏈 {away_team} @ {home_team}")
    print(f"{'='*80}")
    
    # Get narrative analysis
    narrative = await scraper.get_game_narrative(home_team, away_team)
    
    # Actual result
    total = actual_result.get("total_score")
    home_score = actual_result.get("home_score")
    away_score = actual_result.get("away_score")
    
    print(f"\n📊 ACTUAL RESULT:")
    print(f"   Score: {away_team} {away_score}, {home_team} {home_score}")
    print(f"   Total: {total}")
    
    print(f"\n📰 NARRATIVE ANALYSIS:")
    print(f"   Public Lean: {narrative['public_lean']:.0%} {'OVER' if narrative['public_lean'] > 0.5 else 'UNDER'}")
    print(f"   Narrative Strength: {narrative['narrative_strength']:.0%}")
    print(f"   Conspiracy Score: {narrative.get('conspiracy_score', 0):.0%}")
    print(f"   Vegas Bait: {narrative.get('vegas_bait', False)}")
    print(f"   Betting Rec: {narrative.get('betting_recommendation', 'N/A')}")
    
    if narrative.get('storylines'):
        print(f"   Storylines: {', '.join(narrative['storylines'])}")
    
    # Check if recommendation would have been right
    betting_rec = narrative.get('betting_recommendation', '')
    conspiracy = narrative.get('conspiracy_score', 0)
    
    print(f"\n🎯 SYSTEM RECOMMENDATION:")
    if conspiracy > 0.7:
        print(f"   ⚠️  HIGH CONSPIRACY ({conspiracy:.0%})")
        print(f"   Recommendation: {betting_rec}")
    elif conspiracy > 0.5:
        print(f"   🟡 MODERATE CONSPIRACY ({conspiracy:.0%})")
        print(f"   Recommendation: {betting_rec}")
    else:
        print(f"   ✅ LOW CONSPIRACY ({conspiracy:.0%})")
        print(f"   Recommendation: {betting_rec or 'NO CLEAR EDGE'}")
    
    return {
        "game": f"{away_team} @ {home_team}",
        "narrative": narrative,
        "actual": actual_result,
    }


async def analyze_all_completed_games(target_date: date):
    """Analyze all completed games from a date"""
    
    results_file = DATA_DIR / f"game_results_{target_date}.json"
    
    if not results_file.exists():
        print(f"❌ No results file found for {target_date}")
        return
    
    results = json.loads(results_file.read_text())
    
    print(f"\n🔍 ANALYZING COMPLETED GAMES FROM {target_date}")
    print(f"{'='*80}\n")
    
    analyses = []
    
    # Get unique games (results file has duplicate game_ids with different weeks)
    seen_games = set()
    
    async with SimpleNarrativeScraper() as scraper:
        for game_id, result in results.items():
            game_key = f"{result['away_team']}_{result['home_team']}"
            
            if game_key in seen_games:
                continue
            
            seen_games.add(game_key)
            
            analysis = await analyze_completed_game(
                game_id,
                result['home_team'],
                result['away_team'],
                result,
                scraper=scraper,
            )
            
            analyses.append(analysis)
            
            await asyncio.sleep(0.5)  # Rate limit
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY - {len(analyses)} GAMES ANALYZED")
    print(f"{'='*80}")
    
    high_conspiracy = [a for a in analyses if a['narrative'].get('conspiracy_score', 0) > 0.7]
    moderate_conspiracy = [a for a in analyses if 0.5 < a['narrative'].get('conspiracy_score', 0) <= 0.7]
    
    if high_conspiracy:
        print(f"\n🚨 {len(high_conspiracy)} HIGH CONSPIRACY GAMES:")
        for a in high_conspiracy:
            conspiracy = a['narrative'].get('conspiracy_score', 0)
            rec = a['narrative'].get('betting_recommendation', 'N/A')
            print(f"   • {a['game']}: {conspiracy:.0%} conspiracy - {rec}")
    
    if moderate_conspiracy:
        print(f"\n🟡 {len(moderate_conspiracy)} MODERATE CONSPIRACY GAMES:")
        for a in moderate_conspiracy:
            conspiracy = a['narrative'].get('conspiracy_score', 0)
            rec = a['narrative'].get('betting_recommendation', 'N/A')
            print(f"   • {a['game']}: {conspiracy:.0%} conspiracy - {rec}")
    
    # Save analysis
    output_file = DATA_DIR / f"retrospective_analysis_{target_date}.json"
    output_file.write_text(json.dumps([{
        "game": a['game'],
        "narrative": a['narrative'],
        "actual_total": a['actual']['total_score'],
    } for a in analyses], indent=2))
    
    print(f"\n💾 Saved to {output_file}")


async def main():
    import sys
    
    if len(sys.argv) > 1:
        target_date = date.fromisoformat(sys.argv[1])
    else:
        target_date = date.today()
    
    await analyze_all_completed_games(target_date)


if __name__ == "__main__":
    asyncio.run(main())
