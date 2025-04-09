import numpy as np
from sklearn.metrics import f1_score


class AnomalyDetectionEnv:
    """Environment for anomaly detection using reinforcement learning"""

    def __init__(self, X_train, y_train=None, X_test=None, y_test=None,
                 reward_metric='f1', threshold_range=(0.0, 1.0), n_thresholds=10):
        """
        Args:
            X_train: Training features
            y_train: Training labels (optional, for supervised mode)
            X_test: Test features (optional)
            y_test: Test labels (optional)
            reward_metric: Metric to use for rewards ('f1', 'accuracy', etc.)
            threshold_range: Range of thresholds to explore
            n_thresholds: Number of threshold steps
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test if X_test is not None else X_train
        self.y_test = y_test if y_test is not None else y_train

        self.reward_metric = reward_metric
        self.threshold_range = threshold_range
        self.thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

        self.current_threshold_idx = 0
        self.anomaly_scores = None
        self.current_step = 0
        self.max_steps = 100

    def reset(self):
        """Reset environment and return initial state"""
        self.current_threshold_idx = len(self.thresholds) // 2  # start in the middle
        self.current_step = 0

        # initial state consists of current threshold and distribution metrics
        state = self._get_state()
        return state

    def step(self, action):
        """
        Take action and return new state, reward, done flag

        Actions:
        0: Decrease threshold
        1: Keep threshold
        2: Increase threshold
        """
        self.current_step += 1

        # apply action
        if action == 0 and self.current_threshold_idx > 0:
            self.current_threshold_idx -= 1
        elif action == 2 and self.current_threshold_idx < len(self.thresholds) - 1:
            self.current_threshold_idx += 1

        # get new state
        state = self._get_state()

        # calculate reward
        reward = self._calculate_reward()

        # check if episode is done
        done = self.current_step >= self.max_steps

        return state, reward, done, {}

    def _get_state(self):
        """Create state representation"""
        # get current threshold
        threshold = self.thresholds[self.current_threshold_idx]

        # calculate anomaly score distribution features
        if self.anomaly_scores is None:
            # TODO: This would be replaced with your actual anomaly scoring function
            self.anomaly_scores = np.random.rand(len(self.X_test))

        # features about the distribution
        mean_score = np.mean(self.anomaly_scores)
        std_score = np.std(self.anomaly_scores)
        ratio_above = np.mean(self.anomaly_scores > threshold)

        return np.array([threshold, mean_score, std_score, ratio_above])

    def _calculate_reward(self):
        """Calculate reward based on chosen metric"""
        if self.y_test is None:
            # unsupervised case - reward based on threshold properties
            threshold = self.thresholds[self.current_threshold_idx]
            ratio_anomalies = np.mean(self.anomaly_scores > threshold)

            # penalize extreme thresholds that classify everything as normal/anomaly
            if ratio_anomalies < 0.01 or ratio_anomalies > 0.5:
                return -1.0
            return 0.1
        else:
            # supervised case - reward based on classification performance
            threshold = self.thresholds[self.current_threshold_idx]
            y_pred = (self.anomaly_scores > threshold).astype(int)

            if self.reward_metric == 'f1':
                return f1_score(self.y_test, y_pred)
            else:
                # default to accuracy
                return np.mean(y_pred == self.y_test)

    def set_anomaly_scores(self, scores):
        """Set anomaly scores from external model"""
        self.anomaly_scores = scores