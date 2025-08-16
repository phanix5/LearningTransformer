#!/bin/bash
# Script to run Jupyter Lab with the playground notebook in the uv environment

echo "Starting Jupyter Lab in the uv environment..."
echo "You can access the notebook at http://localhost:8888"
echo "Press Ctrl+C to stop the server"

# Run Jupyter Lab with the playground notebook
uv run jupyter lab playground.ipynb
