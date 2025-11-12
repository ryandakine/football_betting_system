#!/bin/bash
# Setup Script for Action Network Selenium Scraper

echo "========================================"
echo "🚀 SELENIUM SCRAPER SETUP"
echo "========================================"
echo ""

# Check Python version
echo "1️⃣ Checking Python version..."
python3 --version
echo ""

# Install dependencies
echo "2️⃣ Installing dependencies..."
echo "   - selenium"
echo "   - webdriver-manager"
echo ""

pip install selenium webdriver-manager

echo ""
echo "✅ Dependencies installed!"
echo ""

# Test Chrome/ChromeDriver
echo "3️⃣ Testing ChromeDriver..."
echo ""

python3 -c "
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

print('   Setting up Chrome...')
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

print('   Testing navigation...')
driver.get('https://www.google.com')
print(f'   Page title: {driver.title}')

driver.quit()
print('   ✅ ChromeDriver works!')
"

echo ""
echo "========================================"
echo "✅ SETUP COMPLETE!"
echo "========================================"
echo ""
echo "📝 NEXT STEPS:"
echo ""
echo "1. Test the scraper:"
echo "   python action_network_selenium_scraper.py --show-browser"
echo ""
echo "2. If it works, run headless:"
echo "   python action_network_selenium_scraper.py --week 11 --save"
echo ""
echo "3. Use the data with trap detector:"
echo "   python trap_detector.py --game 'KC @ BUF' --home-handle 0.75"
echo ""
echo "🎯 PRO TIP:"
echo "   Use Warp AI (warp.dev) to run these commands!"
echo "   Just press Ctrl+` and ask Warp AI to run your workflow."
echo ""
