import numpy as np
import random as rd


class Policy:
    """Base class for RL policies"""

    def select_action(self, state, q_values, training=True):
        """
        Select an action based on state and q_values

        Args:
            state: Current state
            q_values: Q-values for all actions in the current state
            training: Whether the agent is in training mode

        Returns:
            Selected action
        """
        raise NotImplementedError("Subclasses must implement select_action")


class EpsilonGreedyPolicy(Policy):
    """Epsilon-greedy policy for exploration-exploitation tradeoff"""

    def __init__(self, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        """
        Args:
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate at which epsilon decays after each episode
        """
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state, q_values, training=True):
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state (not used directly in epsilon-greedy)
            q_values: Q-values for all actions in the current state
            training: Whether to use exploration

        Returns:
            Selected action index
        """
        if not training:
            # during evaluation, always pick best action
            return np.argmax(q_values)

        # exploration: random action
        if rd.random() < self.epsilon:
            return rd.randrange(len(q_values))

        # exploitation: best action based on Q values
        return np.argmax(q_values)

    def decay_epsilon(self):
        """Decay the exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return self.epsilon


class SoftmaxPolicy(Policy):
    """Softmax policy for weighted exploration based on Q-values"""

    def __init__(self, temperature=1.0, temperature_decay=0.995, temperature_min=0.1):
        """
        Args:
            temperature: Controls the randomness of the policy (higher = more random)
            temperature_decay: Rate at which temperature decays
            temperature_min: Minimum temperature
        """
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min

    def select_action(self, state, q_values, training=True):
        """
        Select action using softmax policy

        Args:
            state: Current state
            q_values: Q-values for all actions in the current state
            training: Whether to use the current temperature

        Returns:
            Selected action index
        """
        if not training:
            return np.argmax(q_values)

        # apply temperature scaling
        scaled_q_values = q_values / self.temperature

        # numerical stability: subtract max value before exponentiation
        exp_q_values = np.exp(scaled_q_values - np.max(scaled_q_values))
        probabilities = exp_q_values / np.sum(exp_q_values)

        # sample action according to the softmax distribution
        return np.random.choice(len(q_values), p=probabilities)

    def decay_temperature(self):
        """Decay the temperature parameter"""
        self.temperature = max(self.temperature_min,
                               self.temperature * self.temperature_decay)
        return self.temperature