#!/bin/bash
# Script to activate the virtual environment
# Usage: source activate_venv.sh

cd "$(dirname "$0")"
source venv/bin/activate
echo "Virtual environment activated!"
echo "Python: $(which python)"
echo "Pip: $(which pip)"

