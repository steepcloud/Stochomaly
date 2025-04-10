import numpy as np
from reinforcement.replay import ReplayBuffer
from reinforcement.policies import EpsilonGreedyPolicy


class DQNAgent:
    """Deep Q-Network agent for reinforcement learning"""

    def __init__(self, state_size, action_size, hidden_size=64,
                 learning_rate=0.001, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 batch_size=64, update_target_every=10,
                 policy=None, memory_size=10000):
        """
        Args:
            state_size: Dimension of state
            action_size: Number of possible actions
            hidden_size: Size of hidden layer
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of epsilon decay
            batch_size: Batch size for training
            update_target_every: Steps between target network updates
            policy: Optional external policy object (if None, creates EpsilonGreedyPolicy)
            memory_size: Size of replay buffer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.step_count = 0

        if policy is not None:
            self.policy = policy
        else:
            self.policy = EpsilonGreedyPolicy(
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay
            )

        # create Q-networks (main and target)
        from nn_core.neural_network import NeuralNetwork

        self.q_network = NeuralNetwork(
            input_size=state_size,
            hidden_size=hidden_size,
            output_size=action_size,
            activation='relu',
            output_activation='linear',
            learning_rate=learning_rate,
            optimizer='adam'
        )

        self.target_network = NeuralNetwork(
            input_size=state_size,
            hidden_size=hidden_size,
            output_size=action_size,
            activation='relu',
            output_activation='linear',
            learning_rate=learning_rate,
            optimizer='adam'
        )

        # copy weights to target network
        self._update_target_network()

        # create replay buffer
        self.replay_buffer = ReplayBuffer(capacity=memory_size)

    @property
    def epsilon(self):
        """Expose epsilon value from policy for backward compatibility"""
        if hasattr(self.policy, 'epsilon'):
            return self.policy.epsilon
        return None

    @epsilon.setter
    def epsilon(self, value):
        """Set epsilon value in policy for backward compatibility"""
        if hasattr(self.policy, 'epsilon'):
            self.policy.epsilon = value

    def _update_target_network(self):
        """Update target network with weights from main network"""
        # copy all network parameters
        params = self.q_network.get_params()
        self.target_network.set_params(params)

    def get_action(self, state, training=True):
        """Select action using policy"""
        state_array = np.array(state).reshape(1, -1)
        q_values = self.q_network.predict(state_array)

        return self.policy.select_action(state, q_values[0], training)

    def train(self, state, action, reward, next_state, done):
        """Train agent with a single experience"""
        # add to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # only train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        # sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # get target Q-values
        next_q_values = self.target_network.predict(next_states)
        max_next_q = np.max(next_q_values, axis=1)

        # calculate target values
        targets = rewards + (1 - dones) * self.discount_factor * max_next_q

        # get current Q-values
        current_q = self.q_network.predict(states)

        # update only the Q-values for the actions taken
        for i, action in enumerate(actions):
            current_q[i, action] = targets[i]

        # train network
        for i in range(len(states)):
            self.q_network.train(states[i].reshape(1, -1), current_q[i].reshape(1, -1))

        # update epsilon for exploration
        self.policy.decay_epsilon()

        # periodically update target network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self._update_target_network()


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent that reduces value overestimation"""

    def __init__(self, state_size, action_size, hidden_size=64,
                 learning_rate=0.001, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 batch_size=64, update_target_every=10,
                 policy=None, memory_size=10000):

        super().__init__(state_size, action_size, hidden_size,
                         learning_rate, discount_factor,
                         epsilon_start, epsilon_end, epsilon_decay,
                         batch_size, update_target_every,
                         policy, memory_size)

    def train(self, state, action, reward, next_state, done):
        """Train agent using Double DQN algorithm"""
        # add to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # only train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        # sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # DOUBLE DQN: Use main network for action selection
        next_actions = np.argmax(self.q_network.predict(next_states), axis=1)

        # use target network for value evaluation
        next_q_values = self.target_network.predict(next_states)

        # get the Q-values for the selected actions
        double_q_values = next_q_values[np.arange(self.batch_size), next_actions]

        # calculate target values
        targets = rewards + (1 - dones) * self.discount_factor * double_q_values

        # get current Q-values
        current_q = self.q_network.predict(states)

        # update only the Q-values for the actions taken
        for i, action in enumerate(actions):
            current_q[i, action] = targets[i]

        # train network
        for i in range(len(states)):
            self.q_network.train(states[i].reshape(1, -1), current_q[i].reshape(1, -1))

        # update epsilon for exploration
        self.policy.decay_epsilon()

        # periodically update target network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self._update_target_network()