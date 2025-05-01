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
        if hasattr(self.policy, "decay_epsilon"):
            self.policy.decay_epsilon()
        elif hasattr(self.policy, "decay_temperature"):
            self.policy.decay_temperature()

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
        if hasattr(self.policy, "decay_epsilon"):
            self.policy.decay_epsilon()
        elif hasattr(self.policy, "decay_temperature"):
            self.policy.decay_temperature()

        # periodically update target network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self._update_target_network()


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN agent with separate value and advantage streams"""

    def __init__(self, state_size, action_size, hidden_size=64,
                 learning_rate=0.001, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 batch_size=64, update_target_every=10,
                 policy=None, memory_size=10000):

        from nn_core.neural_network import NeuralNetwork

        self.value_network = NeuralNetwork(
            input_size=state_size,
            hidden_size=hidden_size,
            output_size=1,  # value function is scalar
            activation='relu',
            output_activation='linear',
            learning_rate=learning_rate,
            optimizer='adam'
        )

        self.advantage_network = NeuralNetwork(
            input_size=state_size,
            hidden_size=hidden_size,
            output_size=action_size,  # one advantage per action
            activation='relu',
            output_activation='linear',
            learning_rate=learning_rate,
            optimizer='adam'
        )

        self.target_value_network = NeuralNetwork(
            input_size=state_size,
            hidden_size=hidden_size,
            output_size=1,
            activation='relu',
            output_activation='linear',
            learning_rate=learning_rate,
            optimizer='adam'
        )

        self.target_advantage_network = NeuralNetwork(
            input_size=state_size,
            hidden_size=hidden_size,
            output_size=action_size,
            activation='relu',
            output_activation='linear',
            learning_rate=learning_rate,
            optimizer='adam'
        )

        # placeholder for standard interface compatibility
        self.q_network = None
        self.target_network = None

        super().__init__(state_size, action_size, hidden_size,
                         learning_rate, discount_factor,
                         epsilon_start, epsilon_end, epsilon_decay,
                         batch_size, update_target_every,
                         policy, memory_size)

        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.step_count = 0

        if policy is not None:
            self.policy = policy
        else:
            from reinforcement.policies import EpsilonGreedyPolicy
            self.policy = EpsilonGreedyPolicy(
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay
            )


        self._update_target_network()

        from reinforcement.replay import ReplayBuffer
        self.replay_buffer = ReplayBuffer(capacity=memory_size)

    def _predict_q_values(self, states, use_target=False):
        """Calculate Q-values using dueling architecture"""
        if use_target:
            values = self.target_value_network.predict(states)
            advantages = self.target_advantage_network.predict(states)
        else:
            values = self.value_network.predict(states)
            advantages = self.advantage_network.predict(states)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = values + (advantages - np.mean(advantages, axis=1, keepdims=True))
        return q_values

    def _update_target_network(self):
        """Update both value and advantage target networks"""
        value_params = self.value_network.get_params()
        self.target_value_network.set_params(value_params)

        advantage_params = self.advantage_network.get_params()
        self.target_advantage_network.set_params(advantage_params)

    def get_action(self, state, training=True):
        """Select action using policy"""
        state_array = np.array(state).reshape(1, -1)
        q_values = self._predict_q_values(state_array)

        return self.policy.select_action(state, q_values[0], training)

    def train(self, state, action, reward, next_state, done):
        """Train agent using Dueling DQN with Double DQN algorithm"""

        self.replay_buffer.add(state, action, reward, next_state, done)

        # only train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        # sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Double DQN: Use main network for action selection
        next_q_values = self._predict_q_values(next_states)
        next_actions = np.argmax(next_q_values, axis=1)

        # use target network for value evaluation
        target_next_q_values = self._predict_q_values(next_states, use_target=True)
        double_q_values = target_next_q_values[np.arange(self.batch_size), next_actions]

        # calculate target values
        targets = rewards + (1 - dones) * self.discount_factor * double_q_values

        # get current Q-values
        current_q = self._predict_q_values(states)

        # create target Q-values by copying current predictions and updating the selected actions
        target_q = np.copy(current_q)
        for i, action in enumerate(actions):
            target_q[i, action] = targets[i]

        # train both networks separately
        for i in range(len(states)):
            state_i = states[i].reshape(1, -1)

            # calculate target value (average of Q-values)
            target_value = np.mean(target_q[i])
            self.value_network.train(state_i, np.array([[target_value]]))

            # calculate target advantages
            target_adv = target_q[i] - target_value
            self.advantage_network.train(state_i, target_adv.reshape(1, -1))

        # update epsilon for exploration
        if hasattr(self.policy, "decay_epsilon"):
            self.policy.decay_epsilon()
        elif hasattr(self.policy, "decay_temperature"):
            self.policy.decay_temperature()

        # periodically update target network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self._update_target_network()


class A2CAgent:
    """Advantage Actor-Critic agent for policy gradient reinforcement learning"""

    def __init__(self, state_size, action_size, hidden_size=64,
                 actor_lr=0.001, critic_lr=0.001, discount_factor=0.99,
                 entropy_coefficient=0.01, max_grad_norm=0.5):
        """
        Args:
            state_size: Dimension of state
            action_size: Number of possible actions
            hidden_size: Size of hidden layer
            actor_lr: Learning rate for actor network
            critic_lr: Learning rate for critic network
            discount_factor: Discount factor for future rewards
            entropy_coefficient: Weight for entropy bonus (encourages exploration)
            max_grad_norm: Maximum norm for gradient clipping
        """
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm

        # actor (policy) network with softmax output
        from nn_core.neural_network import NeuralNetwork
        self.actor = NeuralNetwork(
            input_size=state_size,
            hidden_size=hidden_size,
            output_size=action_size,
            activation='relu',
            output_activation='softmax',  # outputs action probabilities
            learning_rate=actor_lr,
            optimizer='adam'
        )

        # critic (value) network
        self.critic = NeuralNetwork(
            input_size=state_size,
            hidden_size=hidden_size,
            output_size=1,  # value function is scalar
            activation='relu',
            output_activation='linear',
            learning_rate=critic_lr,
            optimizer='adam'
        )

        # episode memory (A2C is on-policy, so we don't use replay buffer)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def get_action(self, state, training=True):
        """Select action based on policy network probabilities"""
        state_array = np.array(state).reshape(1, -1)
        action_probs = self.actor.predict(state_array)[0]

        if training:
            # sample from action probability distribution for exploration
            action = np.random.choice(self.action_size, p=action_probs)
        else:
            # during evaluation, pick the most probable action
            action = np.argmax(action_probs)

        return action

    def train(self, state, action, reward, next_state, done):
        """Store experience for batch training"""
        # store the experience
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

        # train at the end of episode
        if done:
            self._train_episode()
            self._clear_episode_memory()

    def _train_episode(self):
        """Train networks using collected trajectory"""
        if not self.states:
            return

        # convert to numpy arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        next_states = np.array(self.next_states)
        dones = np.array(self.dones)

        # get predicted values for all states
        values = self.critic.predict(states).flatten()
        next_values = self.critic.predict(next_states).flatten()

        # calculate advantages and returns
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        # calculate discounted returns and advantages
        for t in range(len(rewards) - 1, -1, -1):
            # bootstrap from next value if not done
            next_value = 0 if dones[t] else next_values[t]

            # discounted return (for value function target)
            if t == len(rewards) - 1:
                returns[t] = rewards[t] + self.discount_factor * next_value * (1 - dones[t])
            else:
                returns[t] = rewards[t] + self.discount_factor * returns[t + 1] * (1 - dones[t])

            # advantage (for policy gradient)
            advantages[t] = returns[t] - values[t]

        # normalize advantages for stable training
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # train critic (value network)
        for i in range(len(states)):
            state_i = states[i].reshape(1, -1)
            value_target = np.array([[returns[i]]])
            self.critic.train(state_i, value_target)

        # train actor (policy network)
        for i in range(len(states)):
            state_i = states[i].reshape(1, -1)
            action_i = actions[i]
            advantage_i = advantages[i]

            # get current policy probabilities
            action_probs = self.actor.predict(state_i)[0]

            # create target probabilities by adding gradient for the selected action
            target_probs = np.copy(action_probs)

            # policy gradient with entropy bonus
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
            target_probs[action_i] += advantage_i + self.entropy_coefficient * entropy

            # normalize and convert to valid probability distribution
            target_probs = np.clip(target_probs, 1e-6, 1.0 - 1e-6)
            target_probs = target_probs / np.sum(target_probs)

            # update policy network
            self.actor.train(state_i, target_probs.reshape(1, -1))

    def _clear_episode_memory(self):
        """Clear the episode memory after training"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []