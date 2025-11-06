# 🚀 Supabase Integration Guide

## ✅ **Google Sheets → Supabase Migration**

I've cleaned up your workflow by removing all Google Sheets nodes and replaced them with Supabase integration. Here's what changed:

### **🗑️ Removed (Google Sheets):**
- ❌ Daily Performance Sheet
- ❌ Betting Recommendations Sheet
- ❌ Avoid List Sheet
- ❌ SMS Logs Sheet

### **✅ Added (Supabase):**
- ✅ **Supabase Insert Node** - Stores all data in your database
- ✅ **MLflow Tracking** - Experiment tracking
- ✅ **Prometheus Metrics** - Performance monitoring

## 🔧 **Supabase Setup**

### **Step 1: Configure Environment Variables**
Add these to your n8n environment:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

### **Step 2: Create Database Tables**
Run this SQL in your Supabase SQL editor:

```sql
-- Daily Performance Table
CREATE TABLE daily_performance (
  id SERIAL PRIMARY KEY,
  date DATE NOT NULL,
  odds_updated_time TIME,
  workflow_scan_time TIMESTAMP,
  total_games INTEGER,
  win_rate DECIMAL(5,2),
  best_edge DECIMAL(5,2),
  high_value_bets INTEGER,
  medium_value_bets INTEGER,
  recommended_bets INTEGER,
  performance_rating VARCHAR(20),
  created_at TIMESTAMP DEFAULT NOW()
);

-- Betting Opportunities Table
CREATE TABLE betting_opportunities (
  id SERIAL PRIMARY KEY,
  game_id VARCHAR(100),
  team VARCHAR(100),
  game_description TEXT,
  fanduel_odds INTEGER,
  best_other_odds INTEGER,
  best_other_book VARCHAR(100),
  difference INTEGER,
  value_rating VARCHAR(20),
  commence_time TIMESTAMP,
  detected_at TIMESTAMP DEFAULT NOW()
);

-- Avoid List Table
CREATE TABLE avoid_list (
  id SERIAL PRIMARY KEY,
  team VARCHAR(100),
  game_description TEXT,
  fanduel_odds INTEGER,
  better_book VARCHAR(100),
  better_odds INTEGER,
  disadvantage INTEGER,
  reason VARCHAR(100),
  detected_at TIMESTAMP DEFAULT NOW()
);

-- SMS Logs Table
CREATE TABLE sms_logs (
  id SERIAL PRIMARY KEY,
  phone VARCHAR(20),
  message TEXT,
  bet_details JSONB,
  sent_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_daily_performance_date ON daily_performance(date);
CREATE INDEX idx_betting_opportunities_detected_at ON betting_opportunities(detected_at);
CREATE INDEX idx_avoid_list_detected_at ON avoid_list(detected_at);
CREATE INDEX idx_sms_logs_sent_at ON sms_logs(sent_at);
```

### **Step 3: Create RPC Function**
Create this function in Supabase:

```sql
-- Function to insert betting data
CREATE OR REPLACE FUNCTION insert_betting_data(data JSONB)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  result JSONB;
BEGIN
  -- Insert daily performance
  IF data->'daily_performance' IS NOT NULL THEN
    INSERT INTO daily_performance (
      date, odds_updated_time, workflow_scan_time, total_games,
      win_rate, best_edge, high_value_bets, medium_value_bets,
      recommended_bets, performance_rating
    ) VALUES (
      (data->'daily_performance'->>'date')::DATE,
      (data->'daily_performance'->>'odds_updated_time')::TIME,
      (data->'daily_performance'->>'workflow_scan_time')::TIMESTAMP,
      (data->'daily_performance'->>'total_games')::INTEGER,
      (data->'daily_performance'->>'win_rate')::DECIMAL,
      (data->'daily_performance'->>'best_edge')::DECIMAL,
      (data->'daily_performance'->>'high_value_bets')::INTEGER,
      (data->'daily_performance'->>'medium_value_bets')::INTEGER,
      (data->'daily_performance'->>'recommended_bets')::INTEGER,
      data->'daily_performance'->>'performance_rating'
    );
  END IF;

  -- Insert betting opportunities
  IF data->'opportunities' IS NOT NULL THEN
    INSERT INTO betting_opportunities (
      game_id, team, game_description, fanduel_odds,
      best_other_odds, best_other_book, difference, value_rating, commence_time
    )
    SELECT
      opp->>'game_id',
      opp->>'team',
      opp->>'game_description',
      (opp->>'fanduel_odds')::INTEGER,
      (opp->>'best_other_odds')::INTEGER,
      opp->>'best_other_book',
      (opp->>'difference')::INTEGER,
      opp->>'value_rating',
      (opp->>'commence_time')::TIMESTAMP
    FROM jsonb_array_elements(data->'opportunities') AS opp;
  END IF;

  -- Insert avoid list
  IF data->'avoid_list' IS NOT NULL THEN
    INSERT INTO avoid_list (
      team, game_description, fanduel_odds, better_book, better_odds, disadvantage, reason
    )
    SELECT
      avoid->>'team',
      avoid->>'game_description',
      (avoid->>'fanduel_odds')::INTEGER,
      avoid->>'better_book',
      (avoid->>'better_odds')::INTEGER,
      (avoid->>'disadvantage')::INTEGER,
      avoid->>'reason'
    FROM jsonb_array_elements(data->'avoid_list') AS avoid;
  END IF;

  -- Insert SMS logs
  IF data->'sms_messages' IS NOT NULL THEN
    INSERT INTO sms_logs (phone, message, bet_details)
    SELECT
      sms->>'phone',
      sms->>'message',
      sms->'bet_details'
    FROM jsonb_array_elements(data->'sms_messages') AS sms;
  END IF;

  result := jsonb_build_object(
    'status', 'success',
    'message', 'Data inserted successfully',
    'timestamp', NOW()
  );

  RETURN result;
END;
$$;
```

## 📊 **Enhanced Workflow Flow**

### **New Flow:**
```
Schedule Trigger → Time Analysis → Fetch Live Odds → FanDuel Scanner →
Performance Monitor → SMS Filter → Gold Standard Bridge →
Insert to Supabase → MLflow Tracking → Prometheus Metrics
```

### **Data Storage:**
- **Daily Performance** → `daily_performance` table
- **Betting Opportunities** → `betting_opportunities` table
- **Avoid List** → `avoid_list` table
- **SMS Logs** → `sms_logs` table

## 🎯 **Benefits of Supabase Integration**

### **✅ Real-time Database**
- **Instant data storage** - No more Google Sheets delays
- **ACID compliance** - Data integrity guaranteed
- **Automatic backups** - Built-in Supabase backups

### **✅ Better Performance**
- **Faster queries** - SQL vs Google Sheets API
- **Indexed data** - Optimized for betting analysis
- **Real-time subscriptions** - Live data updates

### **✅ Enhanced Analytics**
- **SQL queries** - Complex betting analysis
- **Data relationships** - Link opportunities to performance
- **Historical analysis** - Long-term trend analysis

### **✅ Scalability**
- **Unlimited rows** - No Google Sheets limits
- **Concurrent access** - Multiple users can access data
- **API access** - Programmatic data access

## 🔍 **Query Examples**

### **Daily Performance Summary:**
```sql
SELECT
  date,
  AVG(win_rate) as avg_win_rate,
  SUM(high_value_bets) as total_high_value,
  COUNT(*) as days_analyzed
FROM daily_performance
WHERE date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY date
ORDER BY date DESC;
```

### **Top Performing Teams:**
```sql
SELECT
  team,
  COUNT(*) as opportunities,
  AVG(difference) as avg_edge,
  MAX(difference) as best_edge
FROM betting_opportunities
WHERE detected_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY team
ORDER BY avg_edge DESC
LIMIT 10;
```

### **Value Rating Distribution:**
```sql
SELECT
  value_rating,
  COUNT(*) as count,
  AVG(difference) as avg_edge
FROM betting_opportunities
WHERE detected_at >= CURRENT_DATE - INTERVAL '24 hours'
GROUP BY value_rating
ORDER BY avg_edge DESC;
```

## 🚀 **Next Steps**

1. **Import the cleaned workflow** into n8n
2. **Set up Supabase environment variables**
3. **Create the database tables and function**
4. **Test the integration**
5. **Monitor data flow** via Supabase dashboard

## 🎉 **What You'll Get**

- **Real-time data storage** in Supabase
- **Better performance** than Google Sheets
- **Enhanced analytics** capabilities
- **Scalable architecture** for growth
- **Automatic backups** and data integrity

**Ready to migrate?** Your workflow is now optimized for Supabase with all the monitoring tools integrated!
