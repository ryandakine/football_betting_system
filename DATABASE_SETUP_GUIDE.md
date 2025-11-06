# Database Setup Guide for FanDuel Analysis

## Step 1: Set Up Supabase Database Tables

### Option A: Manual Setup (Recommended)

1. **Go to your Supabase dashboard**
   - Visit: https://supabase.com/dashboard
   - Select your project

2. **Navigate to SQL Editor**
   - Click "SQL Editor" in the left sidebar
   - Click "New Query"

3. **Copy and paste the SQL schema**
   - Open the file: `database/fanduel_analysis_schema.sql`
   - Copy the entire content
   - Paste it into the SQL editor

4. **Execute the SQL**
   - Click "Run" to create all tables
   - You should see success messages

### Option B: Using the Setup Script

1. **Update your aci.env file** with your Supabase credentials:
   ```env
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_ANON_KEY=your-anon-key-here
   ```

2. **Run the setup script**:
   ```bash
   python setup_fanduel_database.py
   ```

## Step 2: Verify Tables Created

After setup, you should have these tables:

- ✅ `fanduel_betting_analysis` - Main analysis results
- ✅ `fanduel_tracking_history` - Historical tracking data
- ✅ `fanduel_game_analysis` - Individual game analysis
- ✅ `bookmaker_performance` - Daily bookmaker performance
- ✅ `daily_summary_stats` - Daily summary statistics

## Step 3: Test Database Connection

You can test the connection by running:
```bash
python -c "
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv('aci.env')
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))
result = supabase.table('fanduel_betting_analysis').select('*').limit(1).execute()
print('✅ Database connection successful!')
"
```

## Troubleshooting

### If you get "table doesn't exist" errors:
1. Make sure you executed the SQL schema
2. Check that you're using the correct database
3. Verify your Supabase credentials

### If you get connection errors:
1. Check your SUPABASE_URL format
2. Verify your SUPABASE_ANON_KEY is correct
3. Make sure your project is active

## Next Steps

Once the database is set up:
1. Configure your n8n workflow with the table names
2. Test the workflow to ensure data is being stored
3. Monitor the results to see your analysis
