"""
Colab/Kaggle Setup Cell
Run this at the start of your notebook
"""

# Mount Google Drive (for checkpointing)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_PATH = '/content/drive/MyDrive/epymarl_coverage'
    import os
    os.makedirs(DRIVE_PATH, exist_ok=True)
    print(f"✓ Google Drive mounted at {DRIVE_PATH}")
    USE_DRIVE = True
except:
    print("Not running on Colab, skipping Drive mount")
    DRIVE_PATH = './checkpoints'
    USE_DRIVE = False
    import os
    os.makedirs(DRIVE_PATH, exist_ok=True)

# Install EPyMARL
import subprocess
import sys

def setup_epymarl():
    """Install EPyMARL and dependencies"""

    print("Installing dependencies...")

    # Install PyTorch with CUDA
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])

    # Install other dependencies
    packages = [
        "numpy", "scipy", "matplotlib", "pyyaml",
        "tensorboard", "sacred", "networkx",
        "gym==0.21.0"  # EPyMARL needs this version
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

    # Clone EPyMARL
    import os
    if not os.path.exists('epymarl'):
        print("Cloning EPyMARL...")
        subprocess.check_call([
            "git", "clone", "-q",
            "https://github.com/uoe-agents/epymarl.git"
        ])
    else:
        print("EPyMARL already exists")

    # Add to path
    sys.path.insert(0, './epymarl/src')

    print("✓ EPyMARL setup complete!")

    return True

# Run setup
if setup_epymarl():
    print("\n" + "="*50)
    print("READY TO TRAIN!")
    print("="*50)
    print(f"Checkpoints will be saved to: {DRIVE_PATH}")
    print("="*50)
