# Background Agents Deployment Guide

## 🚀 Quick Start

This guide will help you deploy the background agents for your MLB betting system using n8n workflows and Supabase.

## 📋 Prerequisites

### 1. Supabase Setup
- ✅ Supabase project created
- ✅ Database schema deployed
- ✅ API keys configured
- ✅ Environment variables set

### 2. n8n Setup
- ✅ n8n instance running
- ✅ Supabase node configured
- ✅ Slack integration (for alerts)
- ✅ API credentials configured

### 3. Required API Keys
```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key

# External APIs
ODDS_API_KEY=your-odds-api-key
TWITTER_BEARER_TOKEN=your-twitter-token
NEWS_API_KEY=your-news-api-key
SLACK_WEBHOOK_URL=your-slack-webhook

# n8n
N8N_WEBHOOK_BASE_URL=https://your-n8n-instance.com
```

## 🔧 Deployment Steps

### Step 1: Deploy n8n Workflows

#### 1.1 Import Real-time Odds Monitor
1. Open your n8n instance
2. Go to Workflows → Import
3. Upload `n8n-workflows/real-time-odds-monitor.json`
4. Configure the following:
   - **Odds API Key**: Set your odds API key
   - **Supabase Connection**: Configure Supabase node
   - **Slack Webhook**: Set up Slack alerts
5. Activate the workflow

#### 1.2 Import Sentiment Analysis Agent
1. Import `n8n-workflows/sentiment-analysis-agent.json`
2. Configure:
   - **Twitter API**: Set bearer token
   - **News API**: Set API key
   - **Supabase Connection**: Configure storage
   - **Slack Alerts**: Set up sentiment alerts
3. Activate the workflow

#### 1.3 Import Performance Monitor
1. Import `n8n-workflows/performance-monitor-agent.json`
2. Configure:
   - **Supabase Connection**: For data retrieval
   - **Slack Alerts**: For performance alerts
   - **Risk Thresholds**: Adjust alert triggers
3. Activate the workflow

### Step 2: Configure Agent Coordinator

#### 2.1 Set Environment Variables
```bash
# Create .env file
cp .env.template .env

# Edit .env with your values
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
N8N_WEBHOOK_BASE_URL=https://your-n8n-instance.com
```

#### 2.2 Update Webhook URLs
Edit `agent_coordinator.py` and update the webhook URLs:

```python
webhook_urls = {
    'odds_monitor_001': 'https://your-n8n-instance.com/webhook/odds-monitor',
    'sentiment_analyzer_001': 'https://your-n8n-instance.com/webhook/sentiment-analyzer',
    'performance_monitor_001': 'https://your-n8n-instance.com/webhook/performance-monitor',
    # ... add other agents
}
```

### Step 3: Deploy Agent Coordinator

#### 3.1 Install Dependencies
```bash
pip install -r supabase_requirements.txt
pip install aiohttp asyncio
```

#### 3.2 Test Configuration
```bash
python supabase_config.py
```

#### 3.3 Start Agent Coordinator
```bash
python agent_coordinator.py
```

### Step 4: Set Up Monitoring

#### 4.1 Create Slack Channels
Create these Slack channels for alerts:
- `#betting-alerts` - High-value opportunities
- `#sentiment-alerts` - Significant sentiment changes
- `#performance-alerts` - Performance warnings
- `#risk-alerts` - Risk management alerts

#### 4.2 Configure Slack Webhooks
1. Go to Slack App settings
2. Create webhook URLs for each channel
3. Add webhook URLs to n8n workflows

## 📊 Agent Configuration

### Agent Schedules

| Agent | Schedule | Purpose |
|-------|----------|---------|
| Odds Monitor | Every 30 seconds | Real-time odds tracking |
| Sentiment Analyzer | Every 15 minutes | Social media sentiment |
| Performance Monitor | Every 5 minutes | Betting performance |
| Risk Manager | Every 10 minutes | Risk assessment |
| Market Analyzer | Every hour | Market analysis |
| Model Trainer | Daily | AI model retraining |

### Alert Thresholds

#### Odds Monitor
- **High Value**: Edge > 5 cents
- **Good Value**: Edge > 2 cents
- **Alert Frequency**: Real-time

#### Sentiment Analyzer
- **High Confidence**: > 70%
- **Strong Sentiment**: > 5 mentions
- **Alert Frequency**: When thresholds met

#### Performance Monitor
- **Low Accuracy**: < 40%
- **Negative ROI**: < -5%
- **Risk Alert**: Consecutive losses > 5

## 🔍 Monitoring & Debugging

### 1. Agent Status Monitoring

#### Check Agent Status
```python
from agent_coordinator import get_agent_status

status = await get_agent_status()
for agent_id, agent in status.items():
    print(f"{agent_id}: {agent.status}")
```

#### System Health Check
```python
from agent_coordinator import get_system_health

health = await get_system_health()
print(f"System Status: {health['system_status']}")
```

### 2. n8n Workflow Monitoring

#### Check Workflow Status
1. Go to n8n dashboard
2. Navigate to Workflows
3. Check execution history
4. Monitor error logs

#### Common Issues
- **Webhook not receiving**: Check n8n URL and firewall
- **API rate limits**: Adjust trigger intervals
- **Supabase errors**: Verify credentials and permissions

### 3. Supabase Monitoring

#### Check Data Flow
```sql
-- Check recent agent activity
SELECT * FROM agent_activity
WHERE created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC;

-- Check odds data
SELECT COUNT(*) FROM odds_data
WHERE created_at > NOW() - INTERVAL '1 hour';

-- Check sentiment data
SELECT * FROM sentiment_data
WHERE created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC;
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Agent Not Triggering
**Symptoms**: Agent status stuck in 'stopped'
**Solutions**:
- Check n8n webhook URL
- Verify agent coordinator is running
- Check Supabase connection

#### 2. High Error Rate
**Symptoms**: Many failed executions
**Solutions**:
- Check API rate limits
- Verify API credentials
- Review error logs in n8n

#### 3. Missing Data
**Symptoms**: No data in Supabase tables
**Solutions**:
- Check n8n workflow execution
- Verify Supabase permissions
- Test data insertion manually

#### 4. Alert Spam
**Symptoms**: Too many Slack alerts
**Solutions**:
- Adjust alert thresholds
- Add rate limiting to alerts
- Filter duplicate alerts

### Debug Commands

#### Test Supabase Connection
```bash
python -c "
from supabase_client import MLBSupabaseClient
import asyncio

async def test():
    client = MLBSupabaseClient()
    result = await client.test_connection()
    print(f'Connection: {result}')

asyncio.run(test())
"
```

#### Test Agent Trigger
```bash
curl -X POST https://your-n8n-instance.com/webhook/odds-monitor \
  -H "Content-Type: application/json" \
  -d '{"test": true}'
```

#### Check Agent Logs
```bash
# Check agent activity in Supabase
python -c "
from supabase_client import MLBSupabaseClient
import asyncio

async def check_logs():
    client = MLBSupabaseClient()
    activity = await client.supabase.table('agent_activity').select('*').limit(10).execute()
    print(activity.data)

asyncio.run(check_logs())
"
```

## 📈 Performance Optimization

### 1. Database Optimization

#### Index Optimization
```sql
-- Add indexes for better performance
CREATE INDEX CONCURRENTLY idx_odds_data_timestamp
ON odds_data(created_at);

CREATE INDEX CONCURRENTLY idx_sentiment_data_timestamp
ON sentiment_data(created_at);

CREATE INDEX CONCURRENTLY idx_agent_activity_timestamp
ON agent_activity(created_at);
```

#### Data Retention
```sql
-- Clean up old data (run daily)
DELETE FROM odds_data
WHERE created_at < NOW() - INTERVAL '30 days';

DELETE FROM sentiment_data
WHERE created_at < NOW() - INTERVAL '7 days';

DELETE FROM agent_activity
WHERE created_at < NOW() - INTERVAL '7 days';
```

### 2. API Rate Limiting

#### Adjust Trigger Intervals
- **Odds Monitor**: 30s → 60s (if rate limited)
- **Sentiment Analyzer**: 15m → 30m
- **Performance Monitor**: 5m → 10m

#### Implement Backoff
```python
# Add exponential backoff for API failures
import time
import random

def api_call_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
```

### 3. Resource Management

#### Memory Optimization
- Limit batch sizes in n8n workflows
- Implement data pagination
- Clean up temporary data

#### CPU Optimization
- Use async operations where possible
- Implement caching for repeated queries
- Optimize data processing algorithms

## 🔄 Maintenance

### Daily Tasks
1. **Check agent status**: Verify all agents are running
2. **Review alerts**: Check for any critical issues
3. **Monitor performance**: Review system health metrics
4. **Clean up logs**: Archive old log files

### Weekly Tasks
1. **Performance review**: Analyze agent performance
2. **Update thresholds**: Adjust alert thresholds based on data
3. **Backup data**: Export important data
4. **Update dependencies**: Check for updates

### Monthly Tasks
1. **System audit**: Review all configurations
2. **Performance optimization**: Implement improvements
3. **Security review**: Check access permissions
4. **Documentation update**: Update this guide

## 🎯 Success Metrics

### Key Performance Indicators

#### Data Quality
- **Odds data freshness**: < 1 minute delay
- **Sentiment accuracy**: > 70% confidence
- **Data completeness**: > 95% success rate

#### System Performance
- **Agent uptime**: > 99% availability
- **Response time**: < 5 seconds for alerts
- **Error rate**: < 1% failure rate

#### Business Impact
- **Opportunity detection**: > 10 high-value bets/day
- **Alert accuracy**: < 5% false positives
- **Performance improvement**: > 5% ROI increase

## 📞 Support

### Getting Help
1. **Check logs**: Review n8n and application logs
2. **Test components**: Isolate and test individual agents
3. **Documentation**: Refer to this guide and README files
4. **Community**: Check n8n and Supabase communities

### Emergency Procedures
1. **Stop all agents**: `python agent_coordinator.py --stop`
2. **Disable alerts**: Comment out Slack nodes in n8n
3. **Check system**: Verify Supabase and n8n are running
4. **Restart gradually**: Start agents one by one

This deployment guide should get your background agents up and running quickly while providing the monitoring and maintenance procedures needed for long-term success.
