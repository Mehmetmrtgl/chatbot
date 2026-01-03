#!/bin/bash
# Script to set up and install requirements in the virtual environment
# Usage: ./setup_venv.sh

cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Setup complete! Virtual environment is ready."
echo "To activate it in the future, run: source activate_venv.sh"
echo "Or: source venv/bin/activate"

