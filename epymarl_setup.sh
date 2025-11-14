#!/bin/bash
# EPyMARL Installation Script for Colab/Kaggle
# Run this at the start of your notebook

set -e  # Exit on error

echo "================================================"
echo "Installing EPyMARL and Dependencies"
echo "================================================"

# Install system dependencies
apt-get update -qq
apt-get install -y -qq git build-essential

# Install Python packages
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q numpy scipy matplotlib pyyaml tensorboard sacred
pip install -q gym==0.21.0  # EPyMARL needs older gym
pip install -q networkx

# Clone EPyMARL
if [ ! -d "epymarl" ]; then
    echo "Cloning EPyMARL..."
    git clone https://github.com/uoe-agents/epymarl.git
    cd epymarl

    # Install EPyMARL dependencies
    pip install -q -e .

    cd ..
else
    echo "EPyMARL already cloned"
fi

# Create necessary directories
mkdir -p results
mkdir -p models
mkdir -p logs
mkdir -p sacred_logs

echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo "EPyMARL location: ./epymarl"
echo "Results directory: ./results"
echo "Models directory: ./models"
