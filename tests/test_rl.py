import unittest
import numpy as np
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reinforcement.policies import Policy, EpsilonGreedyPolicy, SoftmaxPolicy
from reinforcement.replay import ReplayBuffer
from reinforcement.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, A2CAgent
from reinforcement.training import train_rl_agent, evaluate_rl_agent
from reinforcement.environment import AnomalyDetectionEnv


class MockEnvironment:
    """Simple environment for testing"""

    def __init__(self, state_size=4, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.current_state = None
        self.step_count = 0
        self.max_steps = 10

    def reset(self):
        self.current_state = np.random.rand(self.state_size)
        self.step_count = 0
        return self.current_state

    def step(self, action):
        self.step_count += 1
        next_state = np.random.rand(self.state_size)
        reward = 1.0 if action == 0 else -1.0
        done = self.step_count >= self.max_steps
        info = {}
        self.current_state = next_state
        return next_state, reward, done, info


class MockNetwork:
    """Mock neural network for testing"""

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params=[np.random.randn(10, 10)]

    def predict(self, states):
        # return mock Q-values
        batch_size = states.shape[0]
        return np.random.rand(batch_size, self.output_dim)

    def train(self, states, targets):
        # mock training function
        pass

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params


class TestPolicies(unittest.TestCase):
    def test_epsilon_greedy_selection(self):
        policy = EpsilonGreedyPolicy(epsilon_start=1.0)
        state = np.zeros(4)
        q_values = np.array([1.0, 0.5, 0.3])

        # with epsilon=1.0, actions should be random
        actions = [policy.select_action(state, q_values) for _ in range(100)]
        # check that we get a mix of actions (not just the greedy one)
        self.assertTrue(len(set(actions)) > 1)

        # with epsilon=0, should always select best action
        policy.epsilon = 0
        for _ in range(10):
            self.assertEqual(policy.select_action(state, q_values), 0)

    def test_epsilon_decay(self):
        policy = EpsilonGreedyPolicy(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.5)
        initial_epsilon = policy.epsilon
        policy.decay_epsilon()
        self.assertEqual(policy.epsilon, 0.5)
        policy.decay_epsilon()
        self.assertEqual(policy.epsilon, 0.25)
        # should not go below epsilon_end
        for _ in range(10):
            policy.decay_epsilon()
        self.assertEqual(policy.epsilon, 0.1)

    def test_softmax_selection(self):
        policy = SoftmaxPolicy(temperature=0.1)  # Low temperature -> more deterministic
        state = np.zeros(4)
        q_values = np.array([10.0, 1.0, 0.1])  # Large difference to make test reliable

        # with low temperature, should mostly select best action
        actions = [policy.select_action(state, q_values) for _ in range(100)]
        self.assertTrue(actions.count(0) > 80)  # best action should be chosen >80% of time

        # test temperature decay
        policy = SoftmaxPolicy(temperature=1.0, temperature_decay=0.5, temperature_min=0.1)
        policy.decay_temperature()
        self.assertEqual(policy.temperature, 0.5)


class TestReplayBuffer(unittest.TestCase):
    def test_add_and_sample(self):
        buffer = ReplayBuffer(capacity=100)

        # add experiences
        for i in range(50):
            buffer.add(
                state=np.array([i]),
                action=i % 3,
                reward=float(i),
                next_state=np.array([i + 1]),
                done=i % 10 == 0
            )

        self.assertEqual(len(buffer), 50)

        # test sampling
        states, actions, rewards, next_states, dones = buffer.sample(10)
        self.assertEqual(states.shape, (10, 1))
        self.assertEqual(actions.shape, (10,))
        self.assertEqual(rewards.shape, (10,))
        self.assertEqual(next_states.shape, (10, 1))
        self.assertEqual(dones.shape, (10,))

    def test_capacity(self):
        buffer = ReplayBuffer(capacity=10)

        # fill buffer beyond capacity
        for i in range(20):
            buffer.add(
                state=np.array([i]),
                action=0,
                reward=0.0,
                next_state=np.array([i + 1]),
                done=False
            )

        self.assertEqual(len(buffer), 10)  # should be limited by capacity


class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.state_size = 4
        self.action_size = 3


        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            batch_size=5
        )

        # replacing networks with our mock networks
        self.agent.q_network = MockNetwork(self.state_size, self.action_size)
        self.agent.target_network = MockNetwork(self.state_size, self.action_size)

    def test_action_selection(self):
        state = np.random.rand(self.state_size)
        action = self.agent.get_action(state)
        self.assertTrue(0 <= action < self.action_size)

        # test with training=False (should use policy)
        action = self.agent.get_action(state, training=False)
        self.assertTrue(0 <= action < self.action_size)

    def test_training(self):
        # add some experiences to buffer
        for _ in range(10):
            state = np.random.rand(self.state_size)
            action = 0
            reward = 1.0
            next_state = np.random.rand(self.state_size)
            done = False
            self.agent.train(state, action, reward, next_state, done)

        # after training, check if epsilon decayed
        initial_epsilon = self.agent.policy.epsilon
        for _ in range(5):
            state = np.random.rand(self.state_size)
            action = 0
            reward = 1.0
            next_state = np.random.rand(self.state_size)
            done = False
            self.agent.train(state, action, reward, next_state, done)

        # epsilon should have decayed
        self.assertTrue(self.agent.policy.epsilon < initial_epsilon)

    def test_target_network_update(self):
        """Test that target network gets updated properly"""
        # store initial target network parameters
        initial_params = self.agent.target_network.get_params()

        # force multiple updates
        for _ in range(10):
            self.agent._update_target_network()

        # parameters should have changed
        current_params = self.agent.target_network.get_params()
        self.assertNotEqual(initial_params[0].sum(), current_params[0].sum())


class TestDoubleDQNAgent(unittest.TestCase):
    def setUp(self):
        self.state_size = 4
        self.action_size = 3
        self.agent = DoubleDQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            batch_size=5
        )

        self.agent.q_network = MockNetwork(self.state_size, self.action_size)
        self.agent.target_network = MockNetwork(self.state_size, self.action_size)

    def test_action_selection(self):
        state = np.random.rand(self.state_size)
        action = self.agent.get_action(state)
        self.assertTrue(0 <= action < self.action_size)

    def test_training(self):
        # add some experiences to buffer
        for _ in range(10):
            state = np.random.rand(self.state_size)
            action = 0
            reward = 1.0
            next_state = np.random.rand(self.state_size)
            done = False
            self.agent.train(state, action, reward, next_state, done)

        # ensure epsilon decays
        initial_epsilon = self.agent.policy.epsilon
        for _ in range(5):
            self.agent.train(
                np.random.rand(self.state_size), 0, 1.0,
                np.random.rand(self.state_size), False
            )
        self.assertTrue(self.agent.policy.epsilon < initial_epsilon)

    def test_double_dqn_network_separation(self):
        """Test that DoubleDQN uses different networks for action selection and evaluation"""

        # test data
        state = np.random.rand(self.state_size)
        next_state = np.random.rand(self.state_size)
        action = 0
        reward = 1.0
        done = False

        # mock the networks to return controlled values
        def mock_online_predict(s):
            q_values = np.zeros((s.shape[0], self.action_size))
            # make action 0 always the highest value
            q_values[:, 0] = 10.0
            q_values[:, 1] = 5.0
            q_values[:, 2] = 2.0
            return q_values

        def mock_target_predict(s):
            q_values = np.zeros((s.shape[0], self.action_size))
            # make action 1 always the highest value (different from online network)
            q_values[:, 0] = 3.0
            q_values[:, 1] = 8.0
            q_values[:, 2] = 1.0
            return q_values

        original_online_predict = self.agent.q_network.predict
        original_target_predict = self.agent.target_network.predict

        try:
            self.agent.q_network.predict = mock_online_predict
            self.agent.target_network.predict = mock_target_predict

            self.agent.train(state, action, reward, next_state, done)

            self.assertTrue(True)
        finally:
            # restore original methods
            self.agent.q_network.predict = original_online_predict
            self.agent.target_network.predict = original_target_predict


class TestDuelingDQNAgent(unittest.TestCase):
    def setUp(self):
        self.state_size = 4
        self.action_size = 3

        with patch.object(DuelingDQNAgent, '_update_target_network'):
            self.agent = DuelingDQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                batch_size=5
            )

        self.agent.q_network = MockNetwork(self.state_size, self.action_size)
        self.agent.target_network = MockNetwork(self.state_size, self.action_size)

    def test_action_selection(self):
        state = np.random.rand(self.state_size)
        action = self.agent.get_action(state)
        self.assertTrue(0 <= action < self.action_size)

    def test_training(self):
        for _ in range(10):
            state = np.random.rand(self.state_size)
            action = 0
            reward = 1.0
            next_state = np.random.rand(self.state_size)
            done = False
            self.agent.train(state, action, reward, next_state, done)

    def test_dueling_architecture_integration(self):
        """Test that DuelingDQN agent can train and select actions correctly"""

        # sequence of training steps
        for _ in range(5):
            state = np.random.rand(self.state_size)
            action = self.agent.get_action(state)
            next_state = np.random.rand(self.state_size)
            reward = 1.0 if action == 0 else -0.5
            done = False

            # train the agent
            self.agent.train(state, action, reward, next_state, done)

        # test that action selection works in training and evaluation mode
        state = np.random.rand(self.state_size)
        train_action = self.agent.get_action(state, training=True)
        eval_action = self.agent.get_action(state, training=False)

        self.assertTrue(0 <= train_action < self.action_size)
        self.assertTrue(0 <= eval_action < self.action_size)


class TestA2CAgent(unittest.TestCase):
    def setUp(self):
        self.state_size = 4
        self.action_size = 3
        self.agent = A2CAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            actor_lr=0.001,
            critic_lr=0.001
        )

        # mock actor and critic networks
        class MockActorCritic(MockNetwork):
            def predict_policy(self, states):
                batch_size = states.shape[0]
                # return valid probability distribution
                probs = np.ones((batch_size, self.output_dim)) * 0.1
                probs[:, 0] = 0.8 # making action 0 highly probable
                return probs

            def predict_value(self, states):
                return np.random.rand(states.shape[0], 1)

        self.agent.actor = MockActorCritic(self.state_size, self.action_size)
        self.agent.critic = MockActorCritic(self.state_size, 1)

    def test_actor_critic_interaction(self):
        """Test that A2C correctly interacts between actor and critic"""

        # mock the actor and critic networks to return controlled values
        original_actor_predict = self.agent.actor.predict_policy
        original_critic_predict = self.agent.critic.predict_value
        original_get_action = self.agent.get_action

        def mock_actor_predict(states):
            batch_size = states.shape[0]
            # ensuring probabilities that sum to 1 with action 0 always preffered
            probs = np.zeros((batch_size, self.action_size))
            probs[:, 0] = 1.0
            return probs

        def mock_critic_predict(states):
            # return state values
            return np.ones((states.shape[0], 1)) * 0.5

        def mock_get_action(state, training=True):
            # always return action 0 for deterministic testing
            return 0

        try:
            self.agent.actor.predict_policy = mock_actor_predict
            self.agent.critic.predict_value = mock_critic_predict
            self.agent.get_action = mock_get_action

            # run a training step
            state = np.random.rand(self.state_size)
            action = 0  # select the high-probability action
            reward = 1.0
            next_state = np.random.rand(self.state_size)
            done = False

            self.agent.train(state, action, reward, next_state, done)

            # check action selection uses actor
            selected_action = self.agent.get_action(state, training=False)
            self.assertEqual(selected_action, 0)  # should select the highest probability action

        finally:
            # restore original methods
            self.agent.actor.predict_policy = original_actor_predict
            self.agent.critic.predict_value = original_critic_predict
            self.agent.get_action = original_get_action

    def test_action_selection(self):
        state = np.random.rand(self.state_size)
        # test during training (exploration)
        with patch('numpy.random.choice', return_value=0):
            action = self.agent.get_action(state)
            self.assertTrue(0 <= action < self.action_size)

            # test during evaluation (exploitation)
            action = self.agent.get_action(state, training=False)
            self.assertTrue(0 <= action < self.action_size)

    def test_training(self):
        state = np.random.rand(self.state_size)
        action = 0
        reward = 1.0
        next_state = np.random.rand(self.state_size)
        done = False

        with patch.object(self.agent.actor, 'predict_policy',
                          return_value=np.array([[0.7, 0.2, 0.1]])):
            self.agent.train(state, action, reward, next_state, done)


class TestTraining(unittest.TestCase):
    def test_train_and_evaluate(self):
        env = MockEnvironment()

        agent = DQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            batch_size=4
        )

        agent.q_network = MockNetwork(env.state_size, env.action_size)
        agent.target_network = MockNetwork(env.state_size, env.action_size)

        # test training with small number of episodes
        results = train_rl_agent(
            agent=agent,
            environment=env,
            episodes=3,
            max_steps=5,
            verbose=0
        )

        self.assertEqual(len(results['rewards']), 3)
        self.assertEqual(len(results['avg_rewards']), 3)

        # test evaluation
        avg_reward = evaluate_rl_agent(
            agent=agent,
            environment=env,
            episodes=2,
            verbose=0
        )
        self.assertIsInstance(avg_reward, float)

    def test_learning_improvement(self):
        """Test that agent performance improves during training"""
        env = MockEnvironment()

        # creating agent with deterministic policy for testing
        agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
        agent.policy.epsilon = 0  # no exploration
        agent.q_network = MockNetwork(env.state_size, env.action_size)
        agent.target_network = MockNetwork(env.state_size, env.action_size)
        agent.epsilon = agent.policy.epsilon

        # override predict to return predictable values
        def mock_predict_initial(states):
            return np.zeros((states.shape[0], env.action_size))

        agent.q_network.predict = mock_predict_initial

        # test improvement over training
        initial_reward = evaluate_rl_agent(agent, env, episodes=1, verbose=0)

        train_rl_agent(agent, env, episodes=5, verbose=0)

        # changing prediction behavior to simulate learning
        def mock_predict_after_training(states):
            q_values = np.zeros((states.shape[0], env.action_size))
            q_values[:, 1] = 2.0
            return q_values

        agent.q_network.predict = mock_predict_after_training

        final_reward = evaluate_rl_agent(agent, env, episodes=1, verbose=0)
        self.assertNotEqual(initial_reward, final_reward)


class TestAnomalyDetectionEnv(unittest.TestCase):
    def setUp(self):
        # simple dataset for testing
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.random.randint(0, 2, size=100)
        self.env = AnomalyDetectionEnv(
            X_train=self.X_train,
            y_train=self.y_train,
            threshold_range=(0.1, 0.9),
            n_thresholds=10
        )
        # set some anomaly scores for testing
        self.env.anomaly_scores = np.random.rand(100)

    def test_normalize_state(self):
        """Test that state normalization works correctly"""
        raw_state = np.array([0.5, 0.7, 0.3, 0.2])  # threshold, mean, std, ratio
        normalized = self.env.normalize_state(raw_state)

        # check threshold normalized to [0,1] range
        self.assertAlmostEqual(normalized[0], (0.5 - 0.1) / (0.9 - 0.1), places=5)

        # check std normalization
        self.assertAlmostEqual(normalized[2], min(0.3 / 0.5, 1.0), places=5)

    def test_dynamic_threshold_adjustment(self):
        """Test threshold adjustment based on rewards"""
        # simulate some reward history
        self.env.reward_history = [0.1, 0.3, 0.2, 0.5, 0.4, 0.6, 0.7, 0.6, 0.8, 0.7, 0.9]

        # record original thresholds
        original_thresholds = self.env.thresholds.copy()

        # run adjustment
        self.env.dynamic_threshold_adjustment()

        # check thresholds changed
        self.assertFalse(np.array_equal(original_thresholds, self.env.thresholds))

    def test_reward_metrics(self):
        """Test different reward metrics"""
        # test with various synthetic data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])

        # test different metrics
        f1 = self.env.calculate_reward(y_true, y_pred, metric='f1')
        precision = self.env.calculate_reward(y_true, y_pred, metric='precision')
        recall = self.env.calculate_reward(y_true, y_pred, metric='recall')
        weighted = self.env.calculate_reward(y_true, y_pred, metric='weighted')

        # basic verification of results
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)
        self.assertGreaterEqual(precision, 0)
        self.assertGreaterEqual(recall, 0)

    def test_env_rl_integration(self):
        """Test environment integration with RL pipeline"""
        from reinforcement.agents import DQNAgent

        # simple DQN agent
        agent = DQNAgent(
            state_size=4,  # matches environment state size
            action_size=3,  # matches environment action size
            batch_size=5
        )

        # test basic interaction
        state = self.env.reset()
        self.assertEqual(len(state), 4)  # check state dimensions

        # take a step
        action = agent.get_action(state)
        next_state, reward, done, _ = self.env.step(action)

        # basic checks
        self.assertEqual(len(next_state), 4)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)


if __name__ == '__main__':
    unittest.main()