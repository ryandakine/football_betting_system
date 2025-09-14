# 🚀 Complete Setup Guide - MLB Betting System

## 📋 Setup Checklist

### ✅ **1. GitHub Repository Secrets Setup**

**Required Secrets:**
- `DOCKER_USERNAME` - Your Docker Hub username
- `DOCKER_PASSWORD` - Your Docker Hub password/token
- `SNYK_TOKEN` - Snyk API token for security scanning

**Setup Steps:**
1. Go to your GitHub repository → Settings → Secrets and variables → Actions
2. Click "New repository secret" for each required secret
3. Enter the secret name and value
4. Click "Add secret"

**Optional Secrets:**
- `SLACK_WEBHOOK_URL` - For Slack notifications
- `DISCORD_WEBHOOK_URL` - For Discord notifications
- `AWS_ACCESS_KEY_ID` - For AWS deployment
- `AWS_SECRET_ACCESS_KEY` - For AWS deployment

### ✅ **2. ACI.dev API Keys Configuration**

**Run the setup script:**
```bash
python aci_dev_setup.py
```

**Or manually configure:**
1. Create `aci.env` file with your API keys
2. Update the configuration in `aci_integration.py`
3. Set environment variables for your services

**Recommended APIs:**
- **The Odds API** - https://the-odds-api.com/
- **NewsAPI** - https://newsapi.org/
- **Twitter API** - https://developer.twitter.com/
- **Slack Webhook** - For notifications

### ✅ **3. MLflow Server Setup**

**Option 1: Using the batch file (Windows)**
```bash
start_mlflow.bat
```

**Option 2: Manual start**
```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:./mlflow_tracking --default-artifact-root file:./mlflow_tracking
```

**Access MLflow UI:**
- URL: http://localhost:5000
- Features: Experiment tracking, model registry, performance monitoring

### ✅ **4. Monitoring Stack Setup**

**Start the monitoring stack:**
```bash
start_monitoring.bat
```

**Or manually:**
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

**Access URLs:**
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **AlertManager**: http://localhost:9093
- **Node Exporter**: http://localhost:9100
- **cAdvisor**: http://localhost:8080

## 🔧 **Configuration Files Created**

### **Monitoring Configuration:**
- `monitoring/prometheus.yml` - Prometheus configuration
- `monitoring/mlb_betting_rules.yml` - Alerting rules
- `monitoring/alertmanager.yml` - AlertManager configuration
- `monitoring/dashboards/mlb-betting-dashboard.json` - Grafana dashboard
- `monitoring/datasources/prometheus.yml` - Grafana datasource

### **Docker Compose:**
- `docker-compose.monitoring.yml` - Monitoring stack
- `docker-compose.yml` - Main application stack

### **CI/CD Configuration:**
- `.github/workflows/ci-cd.yml` - GitHub Actions pipeline
- `.github/dependabot.yml` - Automated dependency updates

## 🚀 **Quick Start Commands**

### **Start All Services:**
```bash
# 1. Start monitoring stack
start_monitoring.bat

# 2. Start MLflow server
start_mlflow.bat

# 3. Start main application
docker-compose up -d
```

### **Verify Setup:**
```bash
# Check Docker containers
docker ps

# Check MLflow
curl http://localhost:5000

# Check Prometheus
curl http://localhost:9090/-/healthy

# Check Grafana
curl http://localhost:3000/api/health
```

## 📊 **Monitoring Dashboard Setup**

### **Grafana Dashboard:**
1. Open http://localhost:3000
2. Login with admin/admin
3. Import the dashboard from `monitoring/dashboards/mlb-betting-dashboard.json`
4. Configure the Prometheus datasource

### **Dashboard Panels:**
- **API Response Time** - Monitor service performance
- **Request Rate** - Track API usage
- **Error Rate** - Monitor system health
- **Betting Opportunities** - Track betting analysis
- **Model Accuracy** - Monitor ML model performance
- **System Resources** - CPU and memory usage

## 🔔 **Alert Configuration**

### **AlertManager Setup:**
1. Update `monitoring/alertmanager.yml` with your webhook URLs
2. Configure Slack/Discord notifications
3. Set up email alerts if needed

### **Alert Rules:**
- **High Error Rate** - Critical alerts for 5xx errors
- **High Response Time** - Warning for slow API responses
- **Service Down** - Critical alerts for service failures
- **Low Model Accuracy** - Warning for poor ML performance
- **System Resources** - Warnings for high CPU/memory usage

## 🧪 **Testing the Setup**

### **Test CI/CD Pipeline:**
```bash
# Make a small change and push
git add .
git commit -m "test: trigger CI/CD pipeline"
git push
```

### **Test Monitoring:**
```bash
# Generate some test metrics
curl http://localhost:8799/health
curl http://localhost:8801/health
curl http://localhost:8802/health
```

### **Test MLflow:**
```python
# Run this Python script to test MLflow
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test-experiment")

with mlflow.start_run():
    mlflow.log_param("test_param", 42)
    mlflow.log_metric("test_metric", 0.85)
    print("MLflow test successful!")
```

## 🔒 **Security Considerations**

### **API Keys:**
- Store all API keys in GitHub secrets
- Never commit API keys to the repository
- Use environment variables in production
- Rotate keys regularly

### **Network Security:**
- Use HTTPS in production
- Configure firewall rules
- Limit access to monitoring endpoints
- Use authentication for Grafana

### **Data Protection:**
- Encrypt sensitive data
- Use secure connections for databases
- Implement proper access controls
- Regular security audits

## 📈 **Performance Optimization**

### **Monitoring Stack:**
- Adjust Prometheus retention period
- Configure appropriate scrape intervals
- Use persistent volumes for data
- Monitor resource usage

### **MLflow:**
- Use external database for production
- Configure artifact storage
- Set up model registry
- Implement backup strategies

## 🆘 **Troubleshooting**

### **Common Issues:**

**1. Port Conflicts:**
```bash
# Check what's using the ports
netstat -ano | findstr :5000
netstat -ano | findstr :9090
netstat -ano | findstr :3000
```

**2. Docker Issues:**
```bash
# Check Docker status
docker info
docker system df

# Clean up if needed
docker system prune -a
```

**3. MLflow Issues:**
```bash
# Check MLflow logs
mlflow server --host 0.0.0.0 --port 5000 --verbose

# Reset MLflow data
rm -rf mlflow_tracking/
```

**4. Monitoring Issues:**
```bash
# Check container logs
docker-compose -f docker-compose.monitoring.yml logs

# Restart services
docker-compose -f docker-compose.monitoring.yml restart
```

## 📞 **Support Resources**

### **Documentation:**
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

### **Community:**
- [MLflow Community](https://github.com/mlflow/mlflow/discussions)
- [Prometheus Community](https://prometheus.io/community/)
- [Grafana Community](https://community.grafana.com/)

## 🎉 **Success Indicators**

Your setup is complete when:
- ✅ GitHub Actions pipeline runs successfully
- ✅ MLflow UI is accessible at http://localhost:5000
- ✅ Prometheus is collecting metrics
- ✅ Grafana dashboard shows data
- ✅ Alerts are configured and working
- ✅ ACI.dev integration is functional

**Congratulations! Your MLB betting system is now fully equipped with enterprise-grade monitoring, CI/CD, and ML operations!** 🚀
