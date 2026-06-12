#!/bin/bash
# Local MkDocs development server for LLM-Agent-System
# Usage: ./scripts/docs-serve.sh
#
# This starts a local server with hot-reload for editing docs.
# Changes are reflected immediately at http://localhost:8012

set -e

# Change to project root
cd "$(dirname "$0")/.."

# Activate virtual environment
source ~/envs/base/bin/activate

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo "Installing mkdocs and material theme..."
    pip install mkdocs mkdocs-material pymdown-extensions
fi

echo "Starting MkDocs development server..."
echo "View docs at: http://localhost:8012"
echo "Press Ctrl+C to stop"
echo ""

mkdocs serve --dev-addr 0.0.0.0:8012
