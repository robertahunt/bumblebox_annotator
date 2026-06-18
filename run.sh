#!/bin/bash
# Quick launch script for Bee Annotator

cd "$(dirname "$0")"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate bee_annotator

# Check if SAM2 checkpoint exists
if [ -f "checkpoints/sam2_hiera_large.pt" ]; then
    echo "Starting Bee Annotator with SAM2..."
    python main.py --sam2-checkpoint checkpoints/sam2_hiera_large.pt
else
    echo "SAM2 checkpoint not found. Starting without SAM2 support."
    echo "Download checkpoint from: https://github.com/facebookresearch/segment-anything-2"
    echo ""
    python main.py
fi
