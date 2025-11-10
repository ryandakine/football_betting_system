#!/bin/bash
# Initialization script for WNBA Betting System

echo "🏀 WNBA Betting System - Initialization"
echo "============================================================="

# Check Python version
echo "📌 Checking Python version..."
python3 --version

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  IMPORTANT: Edit .env and add your API keys!"
else
    echo "✅ .env file already exists"
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/basketball/wnba
mkdir -p logs

echo ""
echo "============================================================="
echo "✅ Installation Complete!"
echo "============================================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the system: python3 run.py"
echo ""
echo "WNBA Season: May - October"
echo "Best time to use: Regular season (May-September)"
echo ""
echo "For more information, see INSTALL.md and README.md"
