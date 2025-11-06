# 🚀 How to Start n8n Locally

## **Option 1: Using the Batch File (Recommended)**

1. **Run the batch file:**
   ```bash
   start_n8n.bat
   ```

2. **Access n8n:**
   - Open your browser
   - Go to: http://localhost:5678
   - n8n will start automatically

## **Option 2: Manual Start**

### **If n8n is installed globally:**
```bash
n8n start
```

### **If n8n is installed via npm:**
```bash
npx n8n start
```

### **If n8n is installed via Docker:**
```bash
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n
```

## **Option 3: Install n8n (if not installed)**

### **Install globally:**
```bash
npm install n8n -g
```

### **Install locally:**
```bash
npm install n8n
```

## **Access n8n**

Once started, n8n will be available at:
- **URL**: http://localhost:5678
- **Default**: No login required (local mode)

## **Export Your Workflow**

Once n8n is running:

1. **Open n8n** in your browser: http://localhost:5678
2. **Open your workflow** (Enhanced MLB Opportunity Detector)
3. **Click the three dots** (⋮) in the top right
4. **Select "Export"**
5. **Copy the JSON** and paste it here

## **Troubleshooting**

### **Port already in use:**
```bash
# Check what's using port 5678
netstat -ano | findstr :5678

# Kill the process if needed
taskkill /PID [PID_NUMBER] /F
```

### **n8n not found:**
```bash
# Install n8n globally
npm install n8n -g

# Or use npx
npx n8n start
```

### **Permission issues:**
```bash
# Run as administrator (Windows)
# Right-click Command Prompt → Run as administrator
```

## **Next Steps**

Once you export your workflow JSON, I'll help you:
1. **Integrate it with MLflow** for experiment tracking
2. **Connect it to Prometheus** for metrics
3. **Add ACI.dev integrations** for enhanced data sources
4. **Set up automated alerts** via AlertManager
5. **Create comprehensive monitoring** dashboards

---

**Ready to start?** Run `start_n8n.bat` and then export your workflow JSON!
