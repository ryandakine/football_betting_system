#!/usr/bin/env python3
"""
Call AWS Lambda for NFL Predictions
Uses your deployed GGUF models on Lambda
"""

import boto3
import json
from datetime import datetime

# Lambda function name (adjust if yours is different)
LAMBDA_FUNCTION_NAME = "nfl-live-predictions"  # or "enhanced_ai_council_predictions"


def invoke_lambda_predictions(game_data=None):
    """Call Lambda function to get NFL predictions"""

    lambda_client = boto3.client('lambda', region_name='us-east-1')

    # Prepare payload
    payload = {
        "action": "analyze_games",
        "timestamp": datetime.now().isoformat()
    }

    # If specific game data provided
    if game_data:
        payload["game"] = game_data

    print("=" * 70)
    print("🏈 CALLING AWS LAMBDA FOR NFL PREDICTIONS")
    print("=" * 70)
    print(f"\nLambda Function: {LAMBDA_FUNCTION_NAME}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()
    print("Invoking Lambda...")
    print()

    try:
        # Invoke Lambda
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType='RequestResponse',  # Synchronous
            Payload=json.dumps(payload)
        )

        # Parse response
        response_payload = json.loads(response['Payload'].read())

        if response['StatusCode'] == 200:
            print("✅ Lambda invocation successful!")
            print()

            # Parse the body if it's a Lambda proxy response
            if 'body' in response_payload:
                body = json.loads(response_payload['body'])
            else:
                body = response_payload

            # Display results
            display_results(body)

            # Save results
            save_results(body)

            return body
        else:
            print(f"❌ Lambda returned error: {response['StatusCode']}")
            print(response_payload)
            return None

    except Exception as e:
        print(f"❌ Error calling Lambda: {e}")
        import traceback
        traceback.print_exc()
        return None


def display_results(results):
    """Display prediction results"""
    print("=" * 70)
    print("📊 PREDICTION RESULTS")
    print("=" * 70)
    print()

    if 'games_analyzed' in results:
        print(f"Games Analyzed: {results['games_analyzed']}")
        print()

    if 'high_confidence_picks' in results:
        picks = results['high_confidence_picks']
        print(f"🎯 HIGH CONFIDENCE PICKS ({len(picks)}):")
        print()

        for pick in picks:
            print(f"🏈 {pick.get('away_team', 'Team A')} @ {pick.get('home_team', 'Team B')}")
            print(f"   Pick: {pick.get('pick', 'N/A')}")
            print(f"   Confidence: {pick.get('confidence', 0):.1%}")

            if 'spread' in pick:
                print(f"   Spread: {pick['spread']}")
            if 'reasoning' in pick:
                print(f"   Reasoning: {pick['reasoning'][:150]}...")
            print()

    if 'all_predictions' in results:
        print(f"\nTotal Predictions: {len(results['all_predictions'])}")

    print("=" * 70)


def save_results(results):
    """Save results to local file"""
    filename = f"data/lambda_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {filename}")


def main():
    # Check AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS Credentials Valid")
        print(f"   Account: {identity['Account']}")
        print(f"   User: {identity['Arn'].split('/')[-1]}")
        print()
    except Exception as e:
        print(f"❌ AWS credentials not configured: {e}")
        print("\nConfigure with: aws configure")
        return

    # Load tonight's game data (optional)
    game_data = None
    try:
        with open('tonights_game.json', 'r') as f:
            game_data = json.load(f)
            print(f"📋 Using game data: {game_data['away_team']} @ {game_data['home_team']}")
            print()
    except:
        print("ℹ️  No specific game data, Lambda will analyze all live games")
        print()

    # Call Lambda
    results = invoke_lambda_predictions(game_data)

    if results:
        print("\n✅ Done!")
    else:
        print("\n❌ Failed to get predictions")


if __name__ == "__main__":
    main()
