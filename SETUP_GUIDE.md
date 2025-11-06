# MLB Betting System Setup Guide

This guide will help you set up the MLB betting system on your local machine or server.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** - [Download Python](https://www.python.org/downloads/)
- **Node.js 16+** - [Download Node.js](https://nodejs.org/)
- **Git** - [Download Git](https://git-scm.com/)
- **Docker & Docker Compose** - [Download Docker](https://www.docker.com/products/docker-desktop/)

## Step 1: Clone the Repository

```bash
git clone https://github.com/ryandakine/mlb_betting_system.git
cd mlb_betting_system
```

## Step 2: Install Dependencies

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Node.js Dependencies
```bash
npm install
```

## Step 3: Environment Configuration

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```env
# API Keys
YOUTUBE_API_KEY=your_youtube_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
CLAUDE_API_KEY=your_claude_api_key_here

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# External Services
SLACK_WEBHOOK_URL=your_slack_webhook_url
THE_ODDS_API_KEY=your_odds_api_key

# n8n Configuration
N8N_PORT=5678
N8N_RUNNERS_ENABLED=true
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=your_secure_password

# System Configuration
LOG_LEVEL=INFO
DEBUG_MODE=false
ENVIRONMENT=development
```

## Step 4: API Key Setup

### YouTube Data API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable YouTube Data API v3
4. Create credentials (API Key)
5. Add the key to your `.env` file

### Supabase Setup
1. Go to [Supabase](https://supabase.com/)
2. Create a new project
3. Get your project URL and anon key
4. Add them to your `.env` file

### Other APIs (Optional)
- **The Odds API**: For sports betting odds
- **Sports Data APIs**: For game statistics
- **Slack**: For notifications

## Step 5: Database Setup

### Using Supabase (Recommended)
1. Run the database initialization script:
```bash
python db/init_db.py
```

2. Import the schema:
```bash
psql -h your-supabase-host -U postgres -d postgres -f mlb_betting_supabase_schema.sql
```

## Step 6: Start Services

### Start n8n Workflows
```bash
npx n8n start
```

### Start Monitoring Services (Optional)
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### Start Main System
```bash
python main.py
```

## Step 7: Import Workflows

1. Open n8n at http://localhost:5678
2. Import the workflow files from the `n8n-workflows/` directory
3. Configure the workflows with your API keys and settings

## Step 8: Verify Installation

Run the test suite to verify everything is working:

```bash
python -m pytest tests/
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all environment variables are set correctly
   - Check API key permissions and quotas

2. **Database Connection Issues**
   - Verify Supabase credentials
   - Check network connectivity

3. **n8n Workflow Failures**
   - Check n8n execution logs
   - Verify workflow configuration

4. **Python Import Errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility

### Debug Mode

Enable debug logging by setting:
```env
LOG_LEVEL=DEBUG
DEBUG_MODE=true
```

## Security Considerations

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive data
3. **Regularly rotate API keys**
4. **Monitor system access** and logs
5. **Keep dependencies updated**

## Production Deployment

For production deployment:

1. **Use a production database** (not Supabase free tier)
2. **Set up proper monitoring** and alerting
3. **Configure SSL/TLS** for all services
4. **Set up automated backups**
5. **Use a reverse proxy** (nginx/Apache)
6. **Configure firewall rules**

## Support

If you encounter issues:

1. Check the troubleshooting section
2. Review the logs in the `logs/` directory
3. Check GitHub issues for similar problems
4. Create a new issue with detailed information

## Next Steps

After successful setup:

1. **Configure your betting strategies** in the AI Council
2. **Set up monitoring dashboards** in Grafana
3. **Test with small amounts** before full deployment
4. **Monitor system performance** and adjust as needed
5. **Regularly review and update** the system

---

**Remember**: This system is for educational purposes. Always test thoroughly before using with real money.
