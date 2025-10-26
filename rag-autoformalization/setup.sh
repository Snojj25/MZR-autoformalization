#!/bin/bash
# Setup script for RAG-Enhanced Iterative Autoformalization

echo "Setting up RAG-Enhanced Iterative Autoformalization System..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Copy environment file
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file and add your OpenAI API key!"
fi

# Test basic functionality
echo "Running basic tests..."
python test_simple.py

echo ""
echo "Setup complete! To run the full system:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Run: python main.py --mode test"
echo ""
echo "For more options, see README.md"