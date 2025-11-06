import json
import os
import boto3
from datetime import datetime
import requests

s3 = boto3.client('s3')
BUCKET = 'football-betting-system-data'

def lambda_handler(event, context):
    """Fetch odds data and store in S3"""
    
    try:
        # Get API key from environment
        odds_api_key = os.environ.get('ODDS_API_KEY')
        
        if not odds_api_key:
            return {
                'statusCode': 400,
                'body': json.dumps('ODDS_API_KEY not configured')
            }
        
        # Fetch NFL odds
        sport = 'americanfootball_nfl'
        regions = 'us'
        markets = 'h2h,spreads,totals'
        
        url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds/'
        params = {
            'apiKey': odds_api_key,
            'regions': regions,
            'markets': markets,
            'oddsFormat': 'american'
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        odds_data = response.json()
        
        # Store in S3
        timestamp = datetime.utcnow().isoformat()
        key = f'odds/{datetime.utcnow().strftime("%Y-%m-%d")}/{timestamp}.json'
        
        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=json.dumps(odds_data),
            ContentType='application/json'
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Odds data collected successfully',
                's3_key': key,
                'games_found': len(odds_data)
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }
