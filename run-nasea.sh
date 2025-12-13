#!/bin/bash
# Run NASEA with the correct virtual environment
cd "$(dirname "$0")"
source .venv/bin/activate
exec nasea "$@"
