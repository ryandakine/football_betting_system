#!/usr/bin/env python3
import os
import pytest

from unified_nfl_intelligence_system import UnifiedNFLIntelligenceSystem


@pytest.mark.asyncio
async def test_cfb_agent_metrics_present():
    os.environ['AGENT_METRICS_ENABLED'] = '1'
    os.environ['AGENT_CFB_SIMS'] = '200'
    game = {
        'game_id': 'SYR_CFB_SMOKE',
        'home_team': 'Syracuse',
        'away_team': 'Notre Dame',
        'spread': 3.0,
        'conference': 'ACC'
    }
    system = UnifiedNFLIntelligenceSystem(bankroll=10000.0)
    result = await system.run_unified_analysis(game)
    agent = result.get('agent_metrics', {})
    # With sample CSVs, phantom/scandal should be floats present
    assert 'phantom_flag_probability' in agent
    assert 'scandal_score' in agent
    assert agent['phantom_flag_probability'] is not None
    assert agent['scandal_score'] is not None

