#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"

echo "----------------------------------------------------------------------"
echo "Checking Prerequisites..."
echo "----------------------------------------------------------------------"

if ! command -v node &> /dev/null
then
    echo ""
    echo "[ERROR] Node.js is not installed!"
    echo "Node.js is required for the Training UI."
    echo "Please install it using your package manager (e.g., sudo apt install nodejs)"
    echo "or download it from: https://nodejs.org/"
    echo ""
    exit 1
fi

echo "Node.js detected."
echo ""

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo ""
    echo "[ERROR] Python is not installed!"
    echo "Please install Python 3.10 or newer."
    echo ""
    exit 1
fi

echo "Using $PYTHON_CMD..."

if [ ! -d "venv" ]; then
    echo "Creating venv..."
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Failed to create virtual environment."
        echo ""
        exit 1
    fi
else
    echo "Venv already exists."
fi

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo ""
    echo "[ERROR] venv/bin/activate not found!"
    echo ""
    exit 1
fi

echo "----------------------------------------------------------------------"
echo "Installing requirements from requirements.txt..."
echo "----------------------------------------------------------------------"
pip install -r requirements.txt

echo ""
echo "----------------------------------------------------------------------"
echo "Installing UI dependencies (npm install)..."
echo "----------------------------------------------------------------------"
cd training-ui
npm install
cd ..

echo ""
echo "----------------------------------------------------------------------"
echo "Installation Complete!"
echo "----------------------------------------------------------------------"
