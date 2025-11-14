"""
Configuration file for MARL Coverage System
Contains all hyperparameters and settings for training and evaluation.
"""

import torch

# -------------------------
# Environment Configuration
# -------------------------
GRID_SIZE = 20
NUM_AGENTS = 4
SENSOR_RANGE = 5
COMM_RANGE = 8.0
COVERAGE_THRESHOLD = 0.80  # For marking cell 'covered'
COMPLETION_THRESHOLD = 95.0  # Target coverage % for episode end
FOV_DEGREES = 120.0

# -------------------------
# Training Configuration
# -------------------------
MAX_TRAINING_EPISODES = 150
EPISODES_PER_MAP_TYPE = 50
MAX_STEPS_PER_EPISODE = 100
SAVE_INTERVAL = 50

# -------------------------
# Evaluation Configuration
# -------------------------
NUM_EVAL_EPISODES = 10
MAX_EVAL_STEPS = 150
VISUALIZE_TRAINING = True
VISUALIZE_EVAL_FINAL = True
VISUALIZE_EVAL_STEPS = True  # Generate animation ONLY for the LAST eval episode
SAVE_EVAL_ANIMATIONS = True

# -------------------------
# Reward Configuration
# -------------------------
GAMMA_COVERAGE = 20.0  # Reward for coverage
STEP_PENALTY = -0.01  # Penalty per step
ORIENTATION_COST_FACTOR = 0.02  # Penalty for orientation change
INVALID_MOVE_PENALTY = -0.5  # Penalty for hitting wall/obstacle

# -------------------------
# Agent Hyperparameters
# -------------------------
AGENT_CONFIG = {
    'memory_capacity': 20000,
    'batch_size': 128,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.999,
    'learning_rate': 0.0003,
    'use_soft_update': True,
    'soft_update_tau': 0.005,
    'target_update_freq': 1000,
}

# -------------------------
# Network Configuration
# -------------------------
USE_DUELING_DQN = True
INPUT_CHANNELS = 2  # Coverage map + Obstacle map
FEATURE_DIM = 2  # Orientation features (sin, cos)

# -------------------------
# Map Generation Configuration
# -------------------------
NUM_ROOMS_RANGE = (6, 12)
ROOM_SIZE_RANGE = (4, 8)

# Cave map parameters
CAVE_FILL_PROB = 0.48
CAVE_SMOOTHING_ITERATIONS = 5
CAVE_BIRTH_LIMIT = 4
CAVE_DEATH_LIMIT = 3

# Random map parameters
RANDOM_OBSTACLE_DENSITY = 0.15

# -------------------------
# Device Configuration
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Directory Configuration
# -------------------------
MODEL_DIR = "./models_coverage_dqn_final"
RUN_DIR = "./runs/coverage_dqn_experiment_final"
ANIMATION_DIR = "./eval_animations_dqn_final"

# -------------------------
# Random Seeds (for reproducibility)
# -------------------------
RANDOM_SEED = 0
NUMPY_SEED = 0
TORCH_SEED = 0
