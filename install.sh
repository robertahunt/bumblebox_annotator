#!/bin/bash
# Installation script for Bee Annotator

set -e  # Exit on error

echo "===================================="
echo "Bee Annotator Installation"
echo "===================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo ""
echo "Step 1: Creating conda environment..."
conda create -n bee_annotator python=3.10 -y

# Activate environment
echo ""
echo "Step 2: Activating environment..."
eval "$(conda shell.bash hook)"
conda activate bee_annotator

# Install PyTorch with CUDA support (or CPU if no CUDA)
echo ""
echo "Step 3: Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision
fi

# Install main requirements
echo ""
echo "Step 4: Installing main requirements..."
pip install -r requirements.txt

# Install SAM2
echo ""
echo "Step 5: Installing SAM2..."
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Optional: Install detectron2 if on Linux/Mac
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    read -p "Do you want to install detectron2? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing detectron2..."
        pip install 'git+https://github.com/facebookresearch/detectron2.git'
    fi
fi

# Create checkpoints directory
echo ""
echo "Step 6: Creating directories..."
mkdir -p checkpoints

echo ""
echo "===================================="
echo "Installation complete!"
echo "===================================="
echo ""
echo "Next steps:"
echo "1. Download SAM2 checkpoint:"
echo "   cd checkpoints"
echo "   wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
echo ""
echo "2. Activate environment and run:"
echo "   conda activate bee_annotator"
echo "   python main.py --sam2-checkpoint checkpoints/sam2_hiera_large.pt"
echo ""
