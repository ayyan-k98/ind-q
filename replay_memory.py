"""
Replay Memory for Experience Replay in DQN
Stores transitions and samples batches for training.
"""

import random
from collections import deque


class ReplayMemory:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity):
        """
        Initialize replay memory.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        """
        Save a transition to memory.

        Args:
            state: Current state (tuple of tensors)
            action: Action taken
            next_state: Resulting state (tuple of tensors)
            reward: Reward received
            done: Whether episode ended
        """
        # Ensure tensors are detached from graph and on CPU for storage
        s_grid, s_feat = state
        ns_grid, ns_feat = next_state
        state_cpu = (s_grid.detach().cpu(), s_feat.detach().cpu())
        next_state_cpu = (ns_grid.detach().cpu(), ns_feat.detach().cpu())
        self.memory.append((state_cpu, action, next_state_cpu, reward, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            list: Batch of transitions
        """
        # Ensure batch_size doesn't exceed current memory length
        actual_batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, actual_batch_size)

    def __len__(self):
        """Return current size of memory."""
        return len(self.memory)
