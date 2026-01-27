#!/bin/bash
# Setup script for AI Interview Intelligence System

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   AI Interview Intelligence System - Setup Script         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "1️⃣  Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "   ❌ Python 3.8+ required. Please upgrade Python."
    exit 1
fi
echo "   ✅ Python version OK"
echo ""

# Check FFmpeg
echo "2️⃣  Checking FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "   ⚠️  FFmpeg not found. Please install:"
    echo "      Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "      macOS: brew install ffmpeg"
    echo "      Windows: Download from https://ffmpeg.org/download.html"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "   ✅ FFmpeg installed"
fi
echo ""

# Create virtual environment
echo "3️⃣  Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   ⚠️  Virtual environment already exists"
    read -p "   Recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "   ✅ Virtual environment recreated"
    fi
else
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "4️⃣  Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "5️⃣  Upgrading pip..."
pip install --upgrade pip --quiet
echo "   ✅ pip upgraded"
echo ""

# Install dependencies
echo "6️⃣  Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet
if [ $? -eq 0 ]; then
    echo "   ✅ All dependencies installed"
else
    echo "   ❌ Some dependencies failed to install"
    echo "   Try running: pip install -r requirements.txt"
fi
echo ""

# Create necessary directories
echo "7️⃣  Creating directories..."
mkdir -p data/sample_videos
mkdir -p data/models_cache
mkdir -p outputs
mkdir -p models
echo "   ✅ Directories created"
echo ""

# Run system check
echo "8️⃣  Running system check..."
python demo.py
echo ""

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   ✅ Setup Complete!                                      ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Launch web app: streamlit run app.py"
echo "  3. Or run demo: python demo.py"
echo ""
echo "For help, see README.md and docs/"
echo ""
