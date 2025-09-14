# 🚀 Quick Setup Guide: Enhanced MLB Opportunity Detector

## 📋 Prerequisites

Before starting, make sure you have:
- ✅ n8n instance running (local or cloud)
- ✅ Supabase project set up
- ✅ API keys ready

## 🔑 Step 1: Get Your API Keys

### YouTube API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable YouTube Data API v3
4. Create credentials → API Key
5. Copy the key

### Odds API Key
1. Go to [The Odds API](https://the-odds-api.com/)
2. Sign up for free account
3. Get your API key from dashboard
4. Copy the key

### Slack Webhook URL
1. Go to [Slack Apps](https://api.slack.com/apps)
2. Create New App → From scratch
3. Add "Incoming Webhooks" feature
4. Create webhook for channel `#mlb-opportunities`
5. Copy the webhook URL

## 🗄️ Step 2: Set Up Supabase

### Create Tables
Run this SQL in your Supabase SQL editor:

```sql
-- Create sentiment_data table
CREATE TABLE IF NOT EXISTS sentiment_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    source TEXT NOT NULL,
    data JSONB,
    sentiment_summary JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create odds_data table
CREATE TABLE IF NOT EXISTS odds_data (
    id SERIAL PRIMARY KEY,
    game_id TEXT,
    bookmaker TEXT,
    home_team TEXT,
    away_team TEXT,
    commence_time TIMESTAMP,
    moneyline_home INTEGER,
    moneyline_away INTEGER,
    edge_cents INTEGER,
    value_rating TEXT,
    market TEXT,
    source TEXT,
    date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create agent_activity table
CREATE TABLE IF NOT EXISTS agent_activity (
    id SERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    data JSONB,
    status TEXT DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Get Supabase Credentials
1. Go to your Supabase project dashboard
2. Settings → API
3. Copy:
   - Project URL
   - Anon public key
   - Service role key (for admin access)

## ⚙️ Step 3: Configure n8n Environment Variables

In your n8n instance, add these environment variables:

```bash
# API Keys
YOUTUBE_API_KEY=your_youtube_api_key_here
ODDS_API_KEY=your_odds_api_key_here
SLACK_WEBHOOK_URL=your_slack_webhook_url_here

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

## 📥 Step 4: Import the Workflow

1. **Open your n8n instance** in browser
2. **Go to Workflows** → **Import from file**
3. **Select** `n8n-workflows/enhanced-mlb-opportunity-detector.json`
4. **Click Import**

## 🔧 Step 5: Configure the Workflow

### Update Supabase Nodes
1. **Find "Store Sentiment in Supabase"** node
2. **Click to edit**
3. **Update connection:**
   - URL: `{{ $env.SUPABASE_URL }}`
   - API Key: `{{ $env.SUPABASE_ANON_KEY }}`

4. **Repeat for "Store Odds in Supabase"** and **"Log Agent Activity"** nodes

### Update Slack Node
1. **Find "Send Opportunity Alert"** node
2. **Click to edit**
3. **Update:**
   - Channel: `mlb-opportunities`
   - Webhook URL: `{{ $env.SLACK_WEBHOOK_URL }}`

### Test API Connections
1. **Click "Execute Workflow"** button
2. **Check each node** for green checkmarks
3. **Fix any red error nodes**

## 🎯 Step 6: Create Slack Channel

1. **Open Slack**
2. **Create new channel:** `#mlb-opportunities`
3. **Add the webhook** to this channel
4. **Invite team members** who need alerts

## ✅ Step 7: Activate and Test

1. **Toggle workflow to "Active"**
2. **Wait 15 minutes** for first run
3. **Check Slack** for opportunity alerts
4. **Verify Supabase** tables have data

## 🔍 Step 8: Monitor and Optimize

### Check These Locations:
- **Slack:** `#mlb-opportunities` channel
- **Supabase:** Tables for data storage
- **n8n:** Execution history and logs

### Expected Output:
```
🎯 **MLB OPPORTUNITY ALERT!** 🎯

📊 **Quick Summary:**
• Total Opportunities: 5
• High Confidence: 2
• High Value: 1
• Sentiment Aligned: 3
• Contrarian: 1

🔥 **TOP OPPORTUNITIES:**
1. 💎 **Yankees ML** (DraftKings)
   🔥 Confidence: 8.5/10 | ✅ bullish alignment
   💰 Edge: 12¢ | Yankees (-150) vs Red Sox (+130)
   ⏰ 7:05 PM ET
```

## 🚨 Troubleshooting

### Common Issues:

**"API Key Invalid"**
- Check API key format
- Verify API is enabled
- Check usage limits

**"Supabase Connection Failed"**
- Verify URL and keys
- Check RLS policies
- Test connection in Supabase dashboard

**"No Opportunities Found"**
- Check if games are scheduled
- Verify odds API is returning data
- Check YouTube search results

**"Slack Not Sending"**
- Verify webhook URL
- Check channel permissions
- Test webhook manually

## 📞 Support

If you encounter issues:
1. Check n8n execution logs
2. Verify all API keys are working
3. Test each node individually
4. Check Supabase table structure

## 🎉 Success!

Once everything is working, you'll get:
- **Real-time opportunity alerts** every 15 minutes
- **Rich, actionable information** for each bet
- **Complete data storage** in Supabase
- **Professional monitoring** and logging

Your MLB betting system is now **automated and intelligent**! 🚀
