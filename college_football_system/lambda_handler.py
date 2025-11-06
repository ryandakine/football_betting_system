#!/usr/bin/env python3
"""
AWS Lambda Handler for College Football Analysis
=================================================

Handles serverless execution of college football betting analysis.
Can be triggered by EventBridge (scheduled) or API Gateway (on-demand).
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Import analyzer (lazy load to reduce cold start)
_analyzer = None


def get_analyzer():
    """Lazy load analyzer to reduce Lambda cold starts."""
    global _analyzer
    if _analyzer is None:
        from college_football_system.main_analyzer import UnifiedCollegeFootballAnalyzer
        
        # Get bankroll from environment
        bankroll = float(os.getenv('BANKROLL', '50000'))
        _analyzer = UnifiedCollegeFootballAnalyzer(bankroll=bankroll)
        logger.info(f"✅ Analyzer initialized with ${bankroll:,.0f} bankroll")
    
    return _analyzer


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler.
    
    Event types:
    - EventBridge (scheduled): Run full analysis
    - API Gateway: Run analysis with custom parameters
    """
    try:
        logger.info(f"🏈 Lambda invoked: {json.dumps(event)}")
        
        # Determine event source
        event_source = event.get('source', 'unknown')
        
        # Run async analysis
        analyzer = get_analyzer()
        results = asyncio.run(analyzer.run_complete_analysis())
        
        # Format response
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'timestamp': datetime.utcnow().isoformat(),
                'event_source': event_source,
                'results': {
                    'total_games': results.get('total_games', 0),
                    'games_analyzed': results.get('games_analyzed', 0),
                    'high_edge_count': len(results.get('high_edge_games', [])),
                    'parlay_count': len(results.get('parlays', [])),
                    'summary': results.get('summary', {})
                }
            }, default=str),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
        logger.info(f"✅ Analysis complete: {results.get('games_analyzed', 0)} games")
        return response
        
    except Exception as e:
        logger.error(f"❌ Lambda error: {str(e)}", exc_info=True)
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        }


# For local testing
if __name__ == '__main__':
    # Simulate Lambda event
    test_event = {
        'source': 'local-test',
        'detail-type': 'Scheduled Event'
    }
    
    class MockContext:
        aws_request_id = 'local-test-123'
        log_group_name = '/aws/lambda/college-football-analyzer'
        log_stream_name = 'test-stream'
        function_name = 'college-football-analyzer'
        memory_limit_in_mb = 512
        function_version = '$LATEST'
        
        def get_remaining_time_in_millis(self):
            return 300000  # 5 minutes
    
    result = lambda_handler(test_event, MockContext())
    print(json.dumps(result, indent=2))
