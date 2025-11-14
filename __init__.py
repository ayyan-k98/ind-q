"""
MARL Coverage System
Multi-Agent Reinforcement Learning for Coverage Planning

A modular system for training and evaluating multi-agent coverage algorithms
using Deep Q-Networks (DQN) with convolutional architectures.
"""

__version__ = "1.0.0"
__author__ = "MARL Coverage Team"

# Import main components for easier access
from .environment import MARLCoverageEnvironment
from .agent import MARLCoverageAgent
from .data_structures import RobotState, WorldState, CoverageMetrics, CommunicationEvent, Room
from .networks import ConvDQN, DuelingConvDQN
from .replay_memory import ReplayMemory

__all__ = [
    'MARLCoverageEnvironment',
    'MARLCoverageAgent',
    'RobotState',
    'WorldState',
    'CoverageMetrics',
    'CommunicationEvent',
    'Room',
    'ConvDQN',
    'DuelingConvDQN',
    'ReplayMemory'
]
