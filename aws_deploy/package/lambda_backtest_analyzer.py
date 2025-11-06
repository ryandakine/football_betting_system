import json
import os
import boto3
from datetime import datetime, timedelta
from typing import Dict, List
import statistics

s3 = boto3.client('s3')
BUCKET = 'football-betting-system-data'

def calculate_kelly_bet_size(odds: float, win_prob: float, bankroll: float, fraction: float = 0.25) -> float:
    """Calculate Kelly Criterion bet size"""
    if odds > 0:
        decimal_odds = 1 + (odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(odds))
    
    kelly_fraction = (win_prob * decimal_odds - 1) / (decimal_odds - 1)
    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
    
    return bankroll * kelly_fraction * fraction

def analyze_line_movement(historical_odds: List[Dict], current_odds: Dict) -> Dict:
    """Analyze line movement patterns for backtesting"""
    if len(historical_odds) < 2:
        return {'movement': 0, 'direction': 'stable'}
    
    # Get spread movements
    spreads = []
    for odds in historical_odds:
        for bookmaker in odds.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market['key'] == 'spreads':
                    for outcome in market['outcomes']:
                        if 'point' in outcome:
                            spreads.append(outcome['point'])
    
    if len(spreads) > 1:
        movement = spreads[-1] - spreads[0]
        direction = 'favorite_moved' if movement < 0 else 'underdog_moved'
        return {'movement': abs(movement), 'direction': direction}
    
    return {'movement': 0, 'direction': 'stable'}

def find_line_discrepancies(game_odds: Dict) -> List[Dict]:
    """Find profitable line discrepancies across bookmakers"""
    discrepancies = []
    
    bookmakers = game_odds.get('bookmakers', [])
    if len(bookmakers) < 2:
        return discrepancies
    
    # Compare spreads across bookmakers
    spread_points = {}
    for bookmaker in bookmakers:
        for market in bookmaker.get('markets', []):
            if market['key'] == 'spreads':
                for outcome in market['outcomes']:
                    team = outcome['name']
                    point = outcome.get('point', 0)
                    price = outcome.get('price', -110)
                    
                    if team not in spread_points:
                        spread_points[team] = []
                    spread_points[team].append({
                        'bookmaker': bookmaker['title'],
                        'point': point,
                        'price': price
                    })
    
    # Find discrepancies
    for team, points in spread_points.items():
        if len(points) >= 2:
            points_sorted = sorted(points, key=lambda x: x['point'])
            best_line = points_sorted[0]
            worst_line = points_sorted[-1]
            
            if abs(best_line['point'] - worst_line['point']) >= 1.0:
                discrepancies.append({
                    'team': team,
                    'type': 'spread',
                    'best_line': best_line,
                    'worst_line': worst_line,
                    'edge': abs(best_line['point'] - worst_line['point']),
                    'estimated_value': abs(best_line['point'] - worst_line['point']) * 0.02  # 2% per point
                })
    
    return discrepancies

def backtest_strategy(historical_games: List[Dict], strategy: str) -> Dict:
    """Backtest a betting strategy against historical data"""
    bankroll = 10000.0
    bets = []
    
    for game in historical_games:
        # Find line discrepancies
        discrepancies = find_line_discrepancies(game)
        
        if not discrepancies:
            continue
        
        # Filter by strategy
        for disc in discrepancies:
            if disc['estimated_value'] >= 0.05:  # 5% edge minimum
                # Simulate bet
                win_prob = 0.52 + disc['estimated_value']  # Base 52% + edge
                bet_size = calculate_kelly_bet_size(
                    disc['best_line']['price'],
                    win_prob,
                    bankroll,
                    fraction=0.25
                )
                
                if bet_size > 0:
                    bets.append({
                        'game_id': game['id'],
                        'team': disc['team'],
                        'bet_size': bet_size,
                        'odds': disc['best_line']['price'],
                        'edge': disc['estimated_value'],
                        'bookmaker': disc['best_line']['bookmaker']
                    })
    
    # Calculate results
    if not bets:
        return {
            'total_bets': 0,
            'total_wagered': 0,
            'roi': 0,
            'profit': 0
        }
    
    total_wagered = sum(bet['bet_size'] for bet in bets)
    avg_edge = statistics.mean(bet['edge'] for bet in bets)
    estimated_profit = total_wagered * avg_edge
    
    return {
        'total_bets': len(bets),
        'total_wagered': round(total_wagered, 2),
        'estimated_roi': round(avg_edge * 100, 2),
        'estimated_profit': round(estimated_profit, 2),
        'avg_bet_size': round(total_wagered / len(bets), 2),
        'bets': bets[:10]  # Return top 10 bets as examples
    }

def lambda_handler(event, context):
    """Backtest betting strategies using historical odds data"""
    
    try:
        # Get date range for backtesting
        days_back = event.get('days_back', 7)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        # Fetch historical odds from S3
        historical_games = []
        
        for i in range(days_back):
            date = start_date + timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            prefix = f'odds/{date_str}/'
            
            try:
                response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        obj_response = s3.get_object(Bucket=BUCKET, Key=obj['Key'])
                        odds_data = json.loads(obj_response['Body'].read())
                        historical_games.extend(odds_data)
            except Exception as e:
                print(f"Error fetching data for {date_str}: {e}")
                continue
        
        if not historical_games:
            return {
                'statusCode': 404,
                'body': json.dumps('No historical data found')
            }
        
        # Run backtest
        strategy = event.get('strategy', 'line_discrepancy')
        results = backtest_strategy(historical_games, strategy)
        
        # Store results in S3
        timestamp = datetime.utcnow().isoformat()
        result_key = f'backtests/{datetime.utcnow().strftime("%Y-%m-%d")}/{timestamp}.json'
        
        s3.put_object(
            Bucket=BUCKET,
            Key=result_key,
            Body=json.dumps({
                'timestamp': timestamp,
                'strategy': strategy,
                'days_analyzed': days_back,
                'games_analyzed': len(historical_games),
                'results': results
            }, indent=2),
            ContentType='application/json'
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Backtest completed successfully',
                's3_key': result_key,
                'games_analyzed': len(historical_games),
                'total_bets': results['total_bets'],
                'estimated_roi': results['estimated_roi'],
                'estimated_profit': results['estimated_profit']
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }
