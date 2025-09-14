# YouTube Analysis + n8n Setup Guide

## 🎯 **Complete Setup Process**

### **Step 1: Get YouTube API Key**

1. **Go to Google Cloud Console**
   - Visit: https://console.cloud.google.com/
   - Sign in with your Google account

2. **Create/Select Project**
   - Click "Select a project" → "New Project"
   - Name: "MLB Betting System"
   - Click "Create"

3. **Enable YouTube Data API v3**
   - Go to "APIs & Services" → "Library"
   - Search for "YouTube Data API v3"
   - Click "Enable"

4. **Create API Key**
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "API Key"
   - Copy the key (starts with "AIza...")

5. **Restrict API Key** (Recommended)
   - Click on your API key
   - Application restrictions: "HTTP referrers"
   - Add: `https://cloud.n8n.io/*`
   - API restrictions: "Restrict key"
   - Select "YouTube Data API v3"
   - Click "Save"

### **Step 2: Update aci.env**

Edit your `aci.env` file and replace:
```
YOUTUBE_API_KEY=your_youtube_api_key_here
```

With your actual key:
```
YOUTUBE_API_KEY=AIzaSyC...your_actual_key_here
```

### **Step 3: Test YouTube API**

Run the test script:
```bash
.\test_youtube.bat
```

This will:
- ✅ Test your API key
- 📺 Show recent MLB daily picks videos
- 🏆 Analyze team mentions
- 💪 Check confidence indicators
- 📊 Provide sentiment analysis

### **Step 4: Import n8n Workflow**

1. **Open n8n Cloud**
   - Go to your n8n cloud instance
   - Click "Import from file"

2. **Import the Workflow**
   - Select: `n8n-workflows/mlb_youtube_analysis_workflow.json`
   - Click "Import"

3. **Configure Credentials**
   - Click on any YouTube node
   - Click "Add Credential"
   - Select "YouTube API"
   - Enter your API key
   - Name it: "YouTube API"
   - Click "Save"

### **Step 5: Configure Other Credentials**

#### **The Odds API**
- Add environment variable: `THE_ODDS_API_KEY`
- Value: Your Odds API key

#### **OpenAI API** (Optional)
- Add environment variable: `OPENAI_API_KEY`
- Value: Your OpenAI API key

#### **Slack Webhook** (Optional)
- Replace `YOUR_SLACK_WEBHOOK` in the workflow
- With your actual Slack webhook URL

### **Step 6: Test the Complete Workflow**

1. **Click "Execute Workflow"**
2. **Check each node:**
   - ✅ YouTube Search - Daily Picks
   - ✅ YouTube Search - Alternative
   - ✅ Process YouTube Data
   - ✅ Get MLB Odds
   - ✅ Combine YouTube + Odds
   - ✅ AI Analysis
   - ✅ Save Analysis
   - ✅ Format Notification
   - ✅ Send Slack Notification

## 📺 **What the YouTube Analysis Does**

### **Data Collection:**
- 🔍 Searches for "MLB daily picks today"
- 🔍 Searches for "MLB picks for today"
- 📅 Filters videos from last 24 hours
- 🎯 Gets top 10 most relevant videos

### **Content Analysis:**
- 🏆 Identifies most mentioned MLB teams
- 💪 Finds confidence indicators ("lock", "guaranteed", etc.)
- 📊 Analyzes sentiment (positive/negative)
- 🎯 Extracts betting recommendations

### **Integration:**
- 🔗 Combines with live odds data
- 💰 Identifies value bets
- 🤖 Sends to AI for analysis
- 📊 Saves to Supabase database
- 📱 Sends notifications

## 🎯 **Expected Output**

### **YouTube Analysis Results:**
```
🏆 Most Mentioned Teams:
  Yankees: 15 mentions
  Red Sox: 12 mentions
  Dodgers: 8 mentions

💪 Confidence Indicators: 7
  Found: lock, guaranteed, sure thing

✅ Positive Videos: 8
❌ Negative Videos: 2
```

### **Value Bet Identification:**
```
💰 Value Bets Found:
   Boston Red Sox @ New York Yankees
   YouTube Pick: Red Sox
   Confidence: 85.2%
```

## 🚨 **Troubleshooting**

### **YouTube API Issues:**
- **403 Error**: Check API key and quota
- **No videos found**: Try different search terms
- **Quota exceeded**: Wait or upgrade plan

### **n8n Issues:**
- **Credential errors**: Re-add YouTube API credentials
- **Node failures**: Check environment variables
- **Connection errors**: Verify API keys

### **Common Fixes:**
1. **Restart n8n** after adding credentials
2. **Check API quotas** in Google Cloud Console
3. **Verify search terms** are working
4. **Test individual nodes** before running full workflow

## 🎉 **Success Indicators**

✅ **YouTube API test passes**
✅ **n8n workflow imports successfully**
✅ **All nodes execute without errors**
✅ **Data appears in Supabase**
✅ **Notifications are sent**

## 📋 **Files Created**

- `n8n-workflows/mlb_youtube_analysis_workflow.json`
- `test_youtube_api.py`
- `test_youtube.bat`
- `YOUTUBE_N8N_SETUP_GUIDE.md`

Your YouTube analysis system is ready! 🚀
