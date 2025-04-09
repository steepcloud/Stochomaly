import numpy as np
from collections import deque
import random as rd


class ReplayBuffer:
    """Experience replay buffer for RL agents"""

    def __init__(self, capacity=10000):
        """
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random batch of experiences"""
        batch = rd.sample(self.buffer, min(len(self.buffer), batch_size))

        # transpose batch for easier processing
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.bool_)
        )

    def __len__(self):
        return len(self.buffer)