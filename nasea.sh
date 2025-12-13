#!/bin/bash
# NASEA Convenience Wrapper Script
# Makes it easier to run NASEA commands without typing the full path

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Run NASEA CLI with all arguments passed through
"$SCRIPT_DIR/venv/bin/python" -m nasea.cli "$@"
