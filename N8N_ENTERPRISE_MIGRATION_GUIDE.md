# n8n Enterprise Migration Guide

## 🚀 **Overview**

This guide will help you migrate your MLB Betting System from self-hosted n8n to n8n Enterprise for improved reliability, security, and management.

## 📋 **Pre-Migration Checklist**

### **Current System Status**
- ✅ All workflows updated with proper error handling
- ✅ Environment variables documented
- ✅ Database schema ready
- ✅ API keys secured

### **n8n Enterprise Benefits**
- 🔐 **Centralized Environment Variables** - No more hardcoded API keys
- 📊 **Enhanced Monitoring** - Real-time workflow status
- 🔄 **Workflow Versioning** - Track changes and rollbacks
- 👥 **Team Collaboration** - User management and permissions
- 🚀 **Production Reliability** - High availability and scalability

## 🛠️ **Migration Steps**

### **Step 1: Export Current Workflows**

#### **Workflows to Export:**
1. **YouTube Sentiment Analysis** (`mlb_youtube_workflow.json`)
2. **Real-time Odds Monitor** (`n8n-workflows/real-time-odds-monitor.json`)
3. **Enhanced MLB Opportunity Detector** (`n8n-workflows/enhanced-mlb-opportunity-detector.json`)
4. **Performance Monitor Agent** (`n8n-workflows/performance-monitor-agent.json`)
5. **Sentiment Analysis Agent** (`n8n-workflows/sentiment-analysis-agent.json`)

#### **Export Process:**
```bash
# Copy all workflow files to a backup directory
mkdir n8n-enterprise-migration
cp mlb_youtube_workflow.json n8n-enterprise-migration/
cp n8n-workflows/*.json n8n-enterprise-migration/
```

### **Step 2: Prepare Environment Variables**

#### **Required Environment Variables:**
```bash
# API Keys
YOUTUBE_API_KEY=AIzaSyAirGlfovjzmg0xUvwA1VGBFDaFgwfQmYY
GEMINI_API_KEY=AIzaSyAw4NKJvEd8A6Io_75DYGgRm_HXzRXsFDM
OPENAI_API_KEY=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
CLAUDE_API_KEY=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
PERPLEXITY_API_KEY=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

# Supabase Configuration
SUPABASE_URL=https://jufurqxkwbkpoclkeyoi.supabase.co
SUPABASE_ANON_KEY=[TO_BE_ADDED]
SUPABASE_SERVICE_ROLE_KEY=[TO_BE_ADDED]

# External Services
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/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
THE_ODDS_API_KEY=ba91c642121f5bdd0d0357656c7c11d9

# System Configuration
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=10
MIN_CONFIDENCE_THRESHOLD=0.65
```

### **Step 3: n8n Enterprise Setup**

#### **1. Install n8n Enterprise**
```bash
# Follow n8n Enterprise installation guide
# https://docs.n8n.io/hosting/installation/enterprise/
```

#### **2. Configure Environment Variables**
1. Go to **Settings** → **Environment Variables**
2. Add all variables from Step 2
3. Test each variable with a simple workflow

#### **3. Import Workflows**
1. Go to **Workflows** → **Import**
2. Import each workflow file
3. Update any hardcoded values to use environment variables

### **Step 4: Update Workflows for Enterprise**

#### **YouTube Sentiment Analysis Workflow**
```javascript
// Replace hardcoded API key with environment variable
const apiKey = $env.YOUTUBE_API_KEY;
const geminiKey = $env.GEMINI_API_KEY;
const slackWebhook = $env.SLACK_WEBHOOK_URL;
```

#### **Real-time Odds Monitor**
```javascript
// Use environment variables for API keys
const oddsApiKey = $env.THE_ODDS_API_KEY;
const supabaseUrl = $env.SUPABASE_URL;
const supabaseKey = $env.SUPABASE_ANON_KEY;
```

### **Step 5: Configure Monitoring**

#### **Set Up Alerts**
1. **Workflow Failures** - Email/Slack notifications
2. **API Rate Limits** - Monitor usage and alerts
3. **Database Errors** - Supabase connection issues
4. **Performance Metrics** - Response times and throughput

#### **Monitoring Dashboard**
- **Workflow Status** - Real-time execution status
- **API Usage** - Rate limit monitoring
- **Error Rates** - Failure tracking
- **Performance** - Response time metrics

### **Step 6: Test All Workflows**

#### **Test Checklist:**
- [ ] YouTube sentiment analysis runs successfully
- [ ] Odds monitoring captures data
- [ ] AI predictions generate correctly
- [ ] Supabase data storage works
- [ ] Slack notifications send properly
- [ ] Error handling works as expected

#### **Test Commands:**
```bash
# Test YouTube API
curl "https://www.googleapis.com/youtube/v3/search?part=snippet&q=MLB&key=$YOUTUBE_API_KEY"

# Test Supabase connection
curl -X GET "$SUPABASE_URL/rest/v1/sentiment_data" \
  -H "apikey: $SUPABASE_ANON_KEY" \
  -H "Authorization: Bearer $SUPABASE_ANON_KEY"
```

## 🔧 **Post-Migration Configuration**

### **1. Schedule Optimization**
- **YouTube Analysis**: Daily at 7:00 AM
- **Odds Monitoring**: Every 5 minutes during games
- **AI Predictions**: 2 hours before each game
- **Performance Reports**: Weekly summaries

### **2. Security Enhancements**
- **API Key Rotation** - Regular key updates
- **Access Control** - User permissions
- **Audit Logging** - Track all changes
- **Backup Strategy** - Regular workflow backups

### **3. Performance Tuning**
- **Concurrent Executions** - Optimize for your workload
- **Resource Allocation** - CPU and memory limits
- **Database Optimization** - Index and query tuning
- **Caching Strategy** - Reduce API calls

## 📊 **Monitoring and Maintenance**

### **Daily Checks**
1. **Workflow Execution Status**
2. **API Usage and Rate Limits**
3. **Database Performance**
4. **Error Logs and Alerts**

### **Weekly Reviews**
1. **Performance Metrics**
2. **System Reliability**
3. **User Access Logs**
4. **Backup Verification**

### **Monthly Maintenance**
1. **Security Updates**
2. **Performance Optimization**
3. **Capacity Planning**
4. **Documentation Updates**

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Environment Variable Errors**
```bash
# Check if variables are set
echo $YOUTUBE_API_KEY
echo $SUPABASE_URL

# Test in n8n
console.log($env.YOUTUBE_API_KEY);
```

#### **Workflow Import Issues**
1. **Check JSON format** - Validate workflow files
2. **Update node versions** - Ensure compatibility
3. **Test connections** - Verify API endpoints
4. **Check permissions** - User access rights

#### **Performance Issues**
1. **Monitor resource usage** - CPU, memory, disk
2. **Optimize database queries** - Index and caching
3. **Reduce API calls** - Implement caching
4. **Scale resources** - Increase capacity

## 📈 **Success Metrics**

### **Technical Metrics**
- **Uptime**: >99.9%
- **Response Time**: <2 seconds
- **Error Rate**: <1%
- **API Success Rate**: >95%

### **Business Metrics**
- **Workflow Completion**: 100%
- **Data Accuracy**: >95%
- **System Reliability**: >99%
- **User Satisfaction**: High

## 🔄 **Rollback Plan**

### **If Migration Fails:**
1. **Keep self-hosted n8n running**
2. **Export enterprise workflows**
3. **Update self-hosted with improvements**
4. **Test thoroughly before switching**

### **Rollback Steps:**
```bash
# Restore self-hosted n8n
docker-compose up -d

# Import updated workflows
# Test all functionality
# Switch traffic back
```

## 📞 **Support Resources**

### **n8n Enterprise Support**
- **Documentation**: https://docs.n8n.io/
- **Community**: https://community.n8n.io/
- **Support**: Enterprise support portal

### **System Documentation**
- **README.md** - System overview
- **API Documentation** - Service integrations
- **Troubleshooting Guide** - Common issues
- **Performance Guide** - Optimization tips

---

**Migration Status**: Ready for Execution
**Estimated Time**: 4-6 hours
**Risk Level**: Low (with rollback plan)
**Success Criteria**: All workflows running in enterprise environment
