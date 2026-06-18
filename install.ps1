# Installation script for Bee Annotator (Windows)

Write-Host "====================================" -ForegroundColor Green
Write-Host "Bee Annotator Installation" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green

# Check if conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Error: conda not found. Please install Anaconda or Miniconda first." -ForegroundColor Red
    exit 1
}

# Create conda environment
Write-Host "`nStep 1: Creating conda environment..." -ForegroundColor Yellow
conda create -n bee_annotator python=3.10 -y

# Activate environment
Write-Host "`nStep 2: Activating environment..." -ForegroundColor Yellow
conda activate bee_annotator

# Install PyTorch with CUDA support (or CPU if no CUDA)
Write-Host "`nStep 3: Installing PyTorch..." -ForegroundColor Yellow
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "CUDA detected, installing PyTorch with CUDA support..." -ForegroundColor Cyan
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "No CUDA detected, installing CPU-only PyTorch..." -ForegroundColor Cyan
    pip install torch torchvision
}

# Install main requirements
Write-Host "`nStep 4: Installing main requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install SAM2
Write-Host "`nStep 5: Installing SAM2..." -ForegroundColor Yellow
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Create checkpoints directory
Write-Host "`nStep 6: Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path checkpoints | Out-Null

Write-Host "`n====================================" -ForegroundColor Green
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Download SAM2 checkpoint:"
Write-Host "   cd checkpoints"
Write-Host "   Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt' -OutFile 'sam2_hiera_large.pt'"
Write-Host "`n2. Activate environment and run:"
Write-Host "   conda activate bee_annotator"
Write-Host "   python main.py --sam2-checkpoint checkpoints/sam2_hiera_large.pt"
Write-Host ""
