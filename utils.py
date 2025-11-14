"""
Utility functions for MARL Coverage System
Contains helper functions used throughout the system.
"""

import os
import errno
import random
import numpy as np
import torch


def ensure_dir(directory):
    """
    Ensure a directory exists, create it if it doesn't.
    Handles race conditions safely.

    Args:
        directory: Path to the directory
    """
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def set_seeds(seed=0):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def moving_average(data, window_size):
    """
    Compute moving average of data.

    Args:
        data: List or array of values
        window_size: Size of the moving window

    Returns:
        tuple: (smoothed_data, x_ticks) - smoothed values and corresponding x positions
    """
    if not data or len(data) < window_size:
        return data, np.arange(len(data))

    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    x_ticks = np.arange(window_size - 1, len(data))
    return smoothed, x_ticks


def get_latest_model_episode(model_dir, agent_id=0):
    """
    Find the latest model episode saved in the model directory.

    Args:
        model_dir: Directory containing model files
        agent_id: Agent ID to check for (default 0)

    Returns:
        int: Latest episode number, or -1 if not found
    """
    latest_ep = -1
    try:
        if os.path.exists(model_dir):
            files = [f for f in os.listdir(model_dir)
                    if f.startswith(f"agent_{agent_id}_ep") and f.endswith(".pt")]
            epochs = [int(f.split('_ep')[1].split('.pt')[0])
                     for f in files
                     if f.split('_ep')[1].split('.pt')[0].isdigit()]
            if epochs:
                latest_ep = max(epochs)
    except Exception as e:
        print(f"Error finding latest model: {e}")

    return latest_ep
