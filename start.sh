#!/bin/bash
# Start script for Parakeet Question Answerer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if package is installed
if ! python -c "import parakeet_qa" 2>/dev/null; then
    echo "Installing parakeet-qa..."
    pip install --upgrade pip
    pip install -e .
fi

# Check for CUDA
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "CUDA is available"
else
    echo "Warning: CUDA not available, using CPU (will be slower)"
fi

# Run the application
# Options can be passed directly: ./start.sh --model llama3.1:8b
exec parakeet-qa "$@"
