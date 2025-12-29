#!/bin/bash
# Setup script for WheatVision2

set -e

echo "=== WheatVision2 Setup ==="

# Create external directory
mkdir -p external

# Clone SAM if not already cloned
if [ ! -d "external/sam_repo" ]; then
    echo "Cloning SAM repository..."
    git clone https://github.com/facebookresearch/segment-anything.git external/sam_repo
    pip install -e external/sam_repo
    
    echo "Please download SAM checkpoint manually:"
    echo "  cd external/sam_repo"
    echo "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h.pth"
else
    echo "SAM repository already exists"
fi

# Clone SAM2 if not already cloned
if [ ! -d "external/sam2_repo" ]; then
    echo "Cloning SAM2 repository..."
    git clone https://github.com/facebookresearch/sam2.git external/sam2_repo
    pip install -e external/sam2_repo
    
    echo "Downloading SAM2 checkpoints..."
    cd external/sam2_repo/checkpoints
    bash download_ckpts.sh
    cd ../../..
else
    echo "SAM2 repository already exists"
fi

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "Please edit .env to match your configuration"
else
    echo ".env already exists"
fi

# Create exports directory
mkdir -p exports

echo ""
echo "=== Setup Complete ==="
echo "Run the app with: python -m wheatvision.ui.app"
