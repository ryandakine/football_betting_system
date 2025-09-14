# MLB Betting System - Supabase Integration

This guide will help you migrate your MLB betting system from Google Sheets and SQLite to Supabase for cloud-based data storage.

## 🚀 Quick Start

### 1. Set Up Supabase Project

1. Go to [supabase.com](https://supabase.com) and create a new project
2. Note your project URL and API keys from the Settings > API section
3. Run the database schema in your Supabase SQL editor

### 2. Install Dependencies

```bash
pip install -r supabase_requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in your project root:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here

# Optional Settings
SUPABASE_BATCH_SIZE=100
SUPABASE_MAX_RETRIES=3
SUPABASE_RETRY_DELAY=1.0
SUPABASE_TIMEOUT=30
```

### 4. Set Up Database Schema

1. Copy the contents of `mlb_betting_supabase_schema.sql`
2. Go to your Supabase project dashboard
3. Navigate to SQL Editor
4. Paste and run the schema

### 5. Test the Integration

```bash
python supabase_usage_example.py
```

## 📊 Database Schema

The Supabase integration includes the following tables:

### Core Tables
- **ai_predictions** - AI model predictions and analysis
- **ensemble_consensus** - Combined AI predictions and consensus
- **daily_portfolios** - Daily betting portfolios and performance
- **recommendations** - Betting recommendations and opportunities
- **professional_bets** - Professional betting system data
- **unit_bets** - Unit-based betting system data

### Data Tables
- **odds_data** - Odds from various bookmakers
- **sentiment_data** - Sentiment analysis data
- **results** - Game results and outcomes
- **metrics** - Performance metrics and analytics

### Tracking Tables
- **ai_performance** - AI model performance tracking
- **learning_metrics** - Learning system metrics
- **analysis_history** - Analysis session history
- **bets** - Placed bets tracking
- **daily_performance** - Daily performance summaries

## 🔄 Migration from Existing Systems

### Migrate from SQLite

Run the migration script to move your existing data:

```bash
python migrate_to_supabase.py
```

This will migrate data from:
- `data/ultimate_betting_system.db`
- `gold_standard_betting.db`
- `professional_betting_history.db`
- `unit_betting_history.db`
- `db/bets.sqlite`
- JSON files in `data/odds/` and `sentiment/`

### Migrate from Google Sheets

If you're currently using Google Sheets, you can:

1. Export your Google Sheets data as CSV
2. Convert to the appropriate format
3. Use the Supabase client to upload the data

## 💻 Usage Examples

### Basic Usage

```python
from supabase_client import MLBSupabaseClient

# Initialize client
supabase_client = MLBSupabaseClient()

# Save AI predictions
await supabase_client.save_ai_predictions(predictions)

# Save recommendations
await supabase_client.save_recommendations(recommendations)

# Save odds data
await supabase_client.save_odds_data(odds_data)
```

### Replace Existing Database Calls

#### Old (SQLite):
```python
from gold_standard_database import GoldStandardDatabase

db = GoldStandardDatabase()
db.save_recommendations(recommendations)
```

#### New (Supabase):
```python
from supabase_client import MLBSupabaseClient

supabase_client = MLBSupabaseClient()
await supabase_client.save_recommendations(recommendations)
```

#### Old (UltimateDatabaseManager):
```python
from ultimate_database_manager import UltimateDatabaseManager

async with UltimateDatabaseManager() as db:
    await db.save_ai_predictions(predictions)
```

#### New (Supabase):
```python
from supabase_client import MLBSupabaseClient

supabase_client = MLBSupabaseClient()
await supabase_client.save_ai_predictions(predictions)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SUPABASE_URL` | Your Supabase project URL | Yes |
| `SUPABASE_ANON_KEY` | Your Supabase anonymous key | Yes |
| `SUPABASE_SERVICE_ROLE_KEY` | Your Supabase service role key | No |
| `SUPABASE_BATCH_SIZE` | Batch size for operations | No |
| `SUPABASE_MAX_RETRIES` | Maximum retry attempts | No |
| `SUPABASE_RETRY_DELAY` | Delay between retries | No |
| `SUPABASE_TIMEOUT` | Connection timeout | No |

### Configuration Validation

```python
from supabase_config import SupabaseConfig

# Validate configuration
SupabaseConfig.validate_config()

# Print configuration (without sensitive data)
SupabaseConfig.print_config()
```

## 📈 Performance Considerations

### Batch Operations
- Use batch operations for large datasets
- Default batch size is 100 records
- Adjust `SUPABASE_BATCH_SIZE` based on your needs

### Connection Management
- The client handles connection pooling automatically
- Set appropriate timeouts for your use case
- Use retry logic for network issues

### Data Types
- JSON data is automatically handled
- Timestamps are converted to ISO format
- UUIDs are generated automatically

## 🔒 Security

### Row Level Security (RLS)
- RLS is enabled on all tables
- Default policies allow full access
- Customize policies based on your authentication needs

### API Keys
- Use environment variables for API keys
- Never commit API keys to version control
- Use service role key only when necessary

## 🐛 Troubleshooting

### Common Issues

#### Connection Errors
```python
# Test connection
if not await supabase_client.test_connection():
    print("Check your SUPABASE_URL and SUPABASE_ANON_KEY")
```

#### Missing Tables
```python
# Check if tables exist
count = await supabase_client.get_table_count('ai_predictions')
print(f"AI predictions table has {count} records")
```

#### Data Type Issues
- Ensure JSON data is properly formatted
- Convert timestamps to ISO format
- Handle None values appropriately

### Error Handling

```python
try:
    await supabase_client.save_ai_predictions(predictions)
except Exception as e:
    logger.error(f"Failed to save predictions: {e}")
    # Handle error appropriately
```

## 📚 API Reference

### MLBSupabaseClient Methods

#### AI Predictions
- `save_ai_predictions(predictions)` - Save AI predictions
- `get_ai_predictions(run_id, date)` - Get AI predictions

#### Recommendations
- `save_recommendations(recommendations)` - Save recommendations
- `get_recommendations(game_id, date)` - Get recommendations

#### Odds Data
- `save_odds_data(odds_data)` - Save odds data
- `get_odds_data(game_id, date, bookmaker)` - Get odds data

#### Sentiment Data
- `save_sentiment_data(sentiment_data)` - Save sentiment data
- `get_sentiment_data(date, source)` - Get sentiment data

#### Utility Methods
- `test_connection()` - Test Supabase connection
- `get_table_count(table_name)` - Get record count for table

## 🚀 Deployment

### Docker
Add to your Dockerfile:
```dockerfile
# Install Supabase dependencies
COPY supabase_requirements.txt .
RUN pip install -r supabase_requirements.txt

# Set environment variables
ENV SUPABASE_URL=your-url
ENV SUPABASE_ANON_KEY=your-key
```

### Environment Variables
Set environment variables in your deployment platform:
- Heroku: Use Config Vars
- AWS: Use Parameter Store or Secrets Manager
- Docker: Use environment files

## 📊 Monitoring

### Database Metrics
Monitor your Supabase usage:
- Row counts per table
- Storage usage
- API request counts
- Performance metrics

### Application Metrics
Track application performance:
- Save operation success rates
- Response times
- Error rates
- Data consistency

## 🔄 Backup and Recovery

### Supabase Backups
- Supabase provides automatic backups
- Configure backup retention policies
- Test restore procedures

### Data Export
```python
# Export data for backup
predictions = await supabase_client.get_ai_predictions()
with open('backup_predictions.json', 'w') as f:
    json.dump(predictions, f)
```

## 🤝 Support

### Documentation
- [Supabase Documentation](https://supabase.com/docs)
- [Python Client Documentation](https://supabase.com/docs/reference/python)

### Community
- [Supabase Discord](https://discord.supabase.com)
- [GitHub Issues](https://github.com/supabase/supabase-python)

### Migration Support
If you need help with migration:
1. Check the migration logs
2. Verify your Supabase configuration
3. Test with small datasets first
4. Contact support if needed

## 📝 Changelog

### v1.0.0
- Initial Supabase integration
- Complete database schema
- Migration utilities
- Basic client implementation

### Future Enhancements
- Real-time subscriptions
- Advanced querying
- Performance optimizations
- Additional data types
