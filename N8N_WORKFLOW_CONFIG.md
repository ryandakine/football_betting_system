# n8n Workflow Configuration Guide

## Step 2: Configure Your n8n Workflow

### 1. Configure "Save FanDuel Analysis" Node

**Settings:**
- **Operation**: Insert
- **Table**: `fanduel_betting_analysis`

**Columns to Insert:**
```
games_data: {{ $json.games }}
summary_data: {{ $json.summary }}
ai_analysis: {{ $('AI FanDuel Analysis').json.choices[0].message.content }}
created_at: {{ new Date().toISOString() }}
```

### 2. Configure "Save Tracking History" Node

**Settings:**
- **Operation**: Insert
- **Table**: `fanduel_tracking_history`

**Columns to Insert:**
```
tracking_data: {{ $json }}
created_at: {{ new Date().toISOString() }}
```

### 3. Set Up Credentials

#### YouTube API Credentials (if using YouTube node):
1. Click on YouTube node
2. Click "Add Credential"
3. Select "YouTube API"
4. Enter your YouTube API key

#### Supabase Credentials:
1. Click on "Save FanDuel Analysis" node
2. Click "Add Credential"
3. Select "Supabase"
4. Enter:
   - **URL**: Your Supabase project URL
   - **API Key**: Your Supabase anon key

### 4. Set Environment Variables

In your n8n environment, set these variables:
- `THE_ODDS_API_KEY` - Your odds API key
- `OPENAI_API_KEY` - Your OpenAI API key
- `SLACK_WEBHOOK_URL` - Your Slack webhook (optional)

### 5. Test the Workflow

1. **Click "Execute Workflow"**
2. **Check each node** for green checkmarks
3. **Review the data** in each node
4. **Check your Supabase database** for new records

## Expected Data Structure

### Sample Output from "Process FanDuel vs Other Books":
```json
{
  "games": [
    {
      "gameId": "12345",
      "homeTeam": "Yankees",
      "awayTeam": "Red Sox",
      "fanduelOdds": { "outcomes": [{"name": "Yankees", "price": -110}] },
      "bestOdds": {
        "homeTeam": { "bestBook": "draftkings", "bestPrice": -105, "fanduelPrice": -110 }
      },
      "moneyLeftOnTable": {
        "homeTeam": 5,
        "awayTeam": 0
      }
    }
  ],
  "summary": {
    "totalGames": 15,
    "totalMoneyLeftOnTable": 47.50,
    "averageMoneyLeftOnTable": 3.17
  }
}
```

## Troubleshooting

### If nodes show errors:
1. **Check credentials** are properly set
2. **Verify environment variables** are configured
3. **Check table names** match exactly
4. **Ensure Supabase connection** is working

### If data isn't saving:
1. **Check Supabase credentials**
2. **Verify table names** in database
3. **Check column names** match exactly
4. **Review Supabase logs** for errors

## Next Steps

After configuration:
1. Test the workflow end-to-end
2. Check data is being stored correctly
3. Monitor the results
4. Adjust settings as needed
