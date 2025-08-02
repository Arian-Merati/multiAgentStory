#!/bin/bash
# /nfs/user/$USER/multiAgentStory/run.sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Installing dependencies..."
# pip install -r requirements.txt

echo "Running pipeline with cluster's Gemma 27B..."

# Execute the python script with the new arguments
# The paths /models/gemma-27b and /results come from your job.yaml file
python run.py \
    --model_path "/models" \
    --device "cuda" \
    --output_dir "/results"

echo "Script finished. Terminating pod."
# This is a good practice to ensure the pod shuts down and releases resources
kill 1