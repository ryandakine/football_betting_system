# YouTube API Setup for n8n

## 🔑 **Step 1: Get YouTube API Key**

### 1. Go to Google Cloud Console
- Visit: https://console.cloud.google.com/
- Sign in with your Google account

### 2. Create a New Project (or use existing)
- Click "Select a project" → "New Project"
- Name it: "MLB Betting System"
- Click "Create"

### 3. Enable YouTube Data API v3
- Go to "APIs & Services" → "Library"
- Search for "YouTube Data API v3"
- Click on it and press "Enable"

### 4. Create API Key
- Go to "APIs & Services" → "Credentials"
- Click "Create Credentials" → "API Key"
- Copy the API key (starts with "AIza...")

### 5. Restrict the API Key (Recommended)
- Click on the API key you just created
- Under "Application restrictions" select "HTTP referrers"
- Add: `https://cloud.n8n.io/*`
- Under "API restrictions" select "Restrict key"
- Select "YouTube Data API v3"
- Click "Save"

## 🎯 **Step 2: Add YouTube API Key to aci.env**

Edit your `aci.env` file and replace:
```
YOUTUBE_API_KEY=your_youtube_api_key_here
```

With your actual API key:
```
YOUTUBE_API_KEY=AIzaSyC...your_actual_key_here
```

## 📺 **Step 3: YouTube Analysis Strategy**

### What We'll Collect:
1. **Daily Picks Videos**: Search for "MLB daily picks today"
2. **Sentiment Analysis**: Analyze video titles and descriptions
3. **Popular Picks**: Identify most mentioned teams/players
4. **Confidence Levels**: Extract confidence indicators from content

### Search Terms to Use:
- "MLB daily picks today"
- "MLB picks for today"
- "baseball picks today"
- "MLB betting picks"
- "daily baseball picks"

## 🔍 **Step 4: Test YouTube API**

Run this command to test your API key:
```bash
python3 -c "import requests; r = requests.get('https://www.googleapis.com/youtube/v3/search', params={'part':'snippet','q':'MLB daily picks','maxResults':1,'key':'YOUR_API_KEY'}); print('✅ Working' if r.status_code == 200 else '❌ Error')"
```

## 📋 **Step 5: n8n Workflow Nodes**

The workflow will include these YouTube nodes:

1. **YouTube Search Node**: Find daily picks videos
2. **YouTube Video Details Node**: Get video information
3. **Code Node**: Analyze sentiment and extract picks
4. **Data Processing Node**: Combine with odds data
5. **AI Analysis Node**: Send to OpenAI for analysis

## 🎯 **Next Steps**

1. Get your YouTube API key
2. Update aci.env file
3. Test the API connection
4. Import the enhanced n8n workflow
5. Configure the YouTube nodes

Ready to proceed? Let me know when you have your YouTube API key!
