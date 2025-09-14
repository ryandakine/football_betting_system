# n8n Integration Guide for Daily Prediction System

## 🚀 **Quick Setup**

### 1. **Import the Workflow**
1. Open your n8n cloud instance
2. Click "Import from file"
3. Select `n8n-workflows/enhanced_daily_prediction_workflow.json`
4. Click "Import"

### 2. **Configure Credentials**

#### **Supabase Credentials** (Already configured)
- ✅ URL: `https://jufurqxkwbkpoclkeyoi.supabase.co`
- ✅ API Key: Already set in workflow

#### **Slack Webhook** (Optional)
1. Go to your Slack workspace
2. Create a new app or use existing one
3. Enable "Incoming Webhooks"
4. Create a webhook URL
5. Replace `YOUR_SLACK_WEBHOOK` in the workflow

### 3. **Test the Integration**

#### **Step 1: Start the Learning API Server**
```bash
python3 simple_daily_system.py
```

#### **Step 2: Run the n8n Workflow**
1. Click "Execute Workflow" in n8n
2. Check each node for success ✅

## 📊 **What the Workflow Does**

### **Node 1: Manual Trigger**
- Starts the workflow manually

### **Node 2: Get Daily Predictions**
- Calls your local API: `http://localhost:8000/learning/predict`
- Gets predictions for today's games

### **Node 3: Process Predictions**
- Formats predictions for notifications
- Identifies value bets
- Creates summary statistics

### **Node 4: Save to Supabase**
- Saves predictions to your database
- Tracks daily performance

### **Node 5: Send Slack Notification**
- Sends formatted predictions to Slack
- Highlights value bets

### **Node 6: Record Outcomes**
- Records game outcomes for learning
- Updates the ML models

## 🔧 **Customization Options**

### **Add More Data Sources**
You can add nodes for:
- **YouTube API**: Get sentiment from daily picks videos
- **The Odds API**: Get real-time odds
- **Reddit API**: Get community sentiment
- **Twitter API**: Get social media sentiment

### **Add Scheduling**
1. Replace "Manual Trigger" with "Cron" node
2. Set schedule: `0 9 * * *` (daily at 9 AM)

### **Add More Notifications**
- **Email**: Add SMTP node
- **Discord**: Add Discord webhook
- **SMS**: Add Twilio node

## 🎯 **Expected Output**

### **Slack Message Example:**
```
🎯 **Daily MLB Predictions**

💰 **Boston Red Sox @ New York Yankees**
   Predicted: Boston Red Sox
   Confidence: 83.1%
   💰 Value Bet: $0.016

⚾ **Tampa Bay Rays @ Toronto Blue Jays**
   Predicted: Toronto Blue Jays
   Confidence: 81.2%

📊 **Summary**
   Total Games: 5
   Value Bets: 3
   Total Stake: $0.032
```

## 🚨 **Troubleshooting**

### **API Connection Issues**
- Ensure `simple_daily_system.py` is running
- Check `http://localhost:8000/health`

### **Supabase Errors**
- Verify API key is correct
- Check table exists: `fanduel_betting_analysis`

### **Slack Notifications**
- Verify webhook URL is correct
- Check Slack app permissions

## 📈 **Next Steps**

1. **Test the workflow** with sample data
2. **Add real API keys** for live data
3. **Set up scheduling** for automated runs
4. **Monitor performance** and adjust

## 🔗 **Files Created**
- `n8n-workflows/enhanced_daily_prediction_workflow.json`
- `N8N_INTEGRATION_GUIDE.md`

Your n8n integration is ready! 🎉
