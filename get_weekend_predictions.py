#!/usr/bin/env python3
"""
Get NFL Weekend Predictions
Analyzes all Sunday + Monday Night Football games
"""

import boto3
import json
from datetime import datetime, timedelta


def get_weekend_schedule():
    """Get this weekend's NFL schedule"""

    # Days to check (Saturday through Monday)
    days = []
    today = datetime.now()

    # Check next 3 days for games
    for i in range(3):
        day = today + timedelta(days=i)
        days.append(day)

    print(f"Checking for games: {today.strftime('%A, %B %d')} - {days[-1].strftime('%A, %B %d')}")
    return days


def invoke_weekend_predictions():
    """Get predictions for all weekend games"""

    lambda_client = boto3.client('lambda', region_name='us-east-1')

    payload = {
        "action": "analyze_weekend",
        "include_days": ["saturday", "sunday", "monday"],
        "timestamp": datetime.now().isoformat()
    }

    print("\n" + "="*70)
    print("🏈 CALLING LAMBDA FOR WEEKEND PREDICTIONS")
    print("="*70)
    print()

    try:
        response = lambda_client.invoke(
            FunctionName='nfl-live-predictions',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )

        response_payload = json.loads(response['Payload'].read())

        if response['StatusCode'] == 200:
            if 'body' in response_payload:
                body = json.loads(response_payload['body'])
            else:
                body = response_payload

            display_weekend_picks(body)
            save_weekend_picks(body)

            return body
        else:
            print(f"❌ Error: {response['StatusCode']}")
            return None

    except Exception as e:
        print(f"❌ Lambda error: {e}")
        import traceback
        traceback.print_exc()
        return None


def display_weekend_picks(results):
    """Display picks organized by day"""

    print("\n" + "="*70)
    print("🏆 WEEKEND BETTING PICKS")
    print("="*70)
    print()

    if 'high_confidence_picks' in results:
        picks = results['high_confidence_picks']

        if not picks:
            print("No high-confidence picks found.")
            print("Check 'all_predictions' for lower confidence plays.")
            return

        # Organize by time slot
        sunday_early = []
        sunday_late = []
        sunday_night = []
        monday_night = []

        for pick in picks:
            game_time = pick.get('game_time', '')

            if 'monday' in game_time.lower():
                monday_night.append(pick)
            elif 'night' in game_time.lower() or '8:' in game_time or '20:' in game_time:
                sunday_night.append(pick)
            elif '4:' in game_time or '16:' in game_time:
                sunday_late.append(pick)
            else:
                sunday_early.append(pick)

        # Display by time slot
        if sunday_early:
            print("📅 SUNDAY EARLY (1:00 PM ET)")
            print("-" * 70)
            display_picks(sunday_early)

        if sunday_late:
            print("\n📅 SUNDAY LATE (4:00 PM ET)")
            print("-" * 70)
            display_picks(sunday_late)

        if sunday_night:
            print("\n📅 SUNDAY NIGHT FOOTBALL (8:20 PM ET)")
            print("-" * 70)
            display_picks(sunday_night)

        if monday_night:
            print("\n📅 MONDAY NIGHT FOOTBALL (8:15 PM ET)")
            print("-" * 70)
            display_picks(monday_night)

    print("\n" + "="*70)
    print(f"Total Games: {results.get('games_analyzed', 0)}")
    print(f"High Confidence Picks: {len(results.get('high_confidence_picks', []))}")
    print("="*70)


def display_picks(picks):
    """Display individual picks"""
    for i, pick in enumerate(picks, 1):
        print(f"\n{i}. {pick.get('away_team', 'Team A')} @ {pick.get('home_team', 'Team B')}")
        print(f"   🎯 Pick: {pick.get('pick', 'N/A')}")
        print(f"   📊 Confidence: {pick.get('confidence', 0):.1%}")

        if 'spread' in pick:
            print(f"   📈 Spread: {pick['spread']}")
        if 'total' in pick:
            print(f"   🎲 Total: {pick['total']}")
        if 'edge' in pick:
            print(f"   💎 Edge: {pick['edge']:.1%}")
        if 'reasoning' in pick:
            print(f"   💡 {pick['reasoning'][:100]}...")


def save_weekend_picks(results):
    """Save weekend picks to file"""
    filename = f"data/weekend_picks_{datetime.now().strftime('%Y%m%d')}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Saved to: {filename}")


def main():
    print("="*70)
    print("🏈 NFL WEEKEND PREDICTIONS")
    print("="*70)
    print()

    # Check AWS
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS Account: {identity['Account']}")
        print()
    except Exception as e:
        print(f"❌ AWS not configured: {e}")
        print("\nRun: aws configure")
        return

    # Get schedule
    days = get_weekend_schedule()

    # Get predictions
    results = invoke_weekend_predictions()

    if results:
        print("\n✅ Weekend picks ready!")
        print("\nNext steps:")
        print("  1. Review picks above")
        print("  2. Check data/weekend_picks_*.json for details")
        print("  3. Place bets before Sunday 1pm ET!")
    else:
        print("\n❌ Failed to get predictions")


if __name__ == "__main__":
    main()
