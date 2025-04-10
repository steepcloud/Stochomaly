import numpy as np


class AnomalyDetectionEnv:
    """Environment for anomaly detection using reinforcement learning"""

    def __init__(self, X_train, y_train=None, X_test=None, y_test=None,
                 reward_metric='f1', threshold_range=(0.0, 1.0), n_thresholds=10,
                 use_dynamic_thresholds=False, adjustment_frequency=20):
        """
        Args:
            X_train: Training features
            y_train: Training labels (optional, for supervised mode)
            X_test: Test features (optional)
            y_test: Test labels (optional)
            reward_metric: Metric to use for rewards ('f1', 'accuracy', etc.)
            threshold_range: Range of thresholds to explore
            n_thresholds: Number of threshold steps
            use_dynamic_thresholds: Whether to dynamically adjust thresholds
            adjustment_frequency: Frequency of dynamic threshold adjustments (steps)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test if X_test is not None else X_train
        self.y_test = y_test if y_test is not None else y_train

        self.reward_metric = reward_metric
        self.threshold_range = threshold_range
        self.thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

        self.use_dynamic_thresholds = use_dynamic_thresholds
        self.adjustment_frequency = adjustment_frequency

        self.current_threshold_idx = 0
        self.anomaly_scores = None
        self.current_step = 0
        self.max_steps = 100

    def reset(self):
        """Reset environment and return initial state"""
        self.current_threshold_idx = len(self.thresholds) // 2  # start in the middle
        self.current_step = 0
        self.reward_history = []

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

        # track rewards for threshold optimization
        if not hasattr(self, 'reward_history'):
            self.reward_history = []

        # apply action
        if action == 0 and self.current_threshold_idx > 0:
            self.current_threshold_idx -= 1
        elif action == 2 and self.current_threshold_idx < len(self.thresholds) - 1:
            self.current_threshold_idx += 1

        # get new state (normalized)
        state = self._get_state()

        # calculate reward
        reward = self._calculate_reward()

        # store reward history
        self.reward_history.append(reward)

        # check if episode is done
        done = self.current_step >= self.max_steps

        # periodically adjust thresholds
        if self.use_dynamic_thresholds and self.current_step % self.adjustment_frequency == 0 and self.current_step > 0:
            self.dynamic_threshold_adjustment()

        return state, reward, done, {}

    def _get_state(self):
        """Create state representation with normalization"""
        # original state calculation
        threshold = self.thresholds[self.current_threshold_idx]

        # calculate anomaly score distribution features
        if self.anomaly_scores is None:
            self.anomaly_scores = np.random.rand(len(self.X_test))

        # features about the distribution
        mean_score = np.mean(self.anomaly_scores)
        std_score = np.std(self.anomaly_scores)
        ratio_above = np.mean(self.anomaly_scores > threshold)

        raw_state = np.array([threshold, mean_score, std_score, ratio_above])

        return self.normalize_state(raw_state)

    def get_state(self):
        """Public method to get the current state representation"""
        return self._get_state()

    def _calculate_reward(self):
        """Calculate reward based on chosen metric"""

        threshold = self.thresholds[self.current_threshold_idx]

        if self.y_test is None:
            # unsupervised case - enhanced reward based on threshold properties
            ratio_anomalies = np.mean(self.anomaly_scores > threshold)

            # penalize extreme thresholds that classify everything as normal/anomaly
            if ratio_anomalies < 0.001: # almost no anomalies
                return -2.0
            elif ratio_anomalies < 0.01: # very few anomalies
                return -1.0
            elif ratio_anomalies > 0.5: # too many anomalies
                return -1.0
            elif ratio_anomalies > 0.3: # high proportion of anomalies
                return -0.5

            # reward stability in reasonable detection range (typically 1-10%)
            if 0.01 <= ratio_anomalies <= 0.1:
                return 0.5

            return 0.1
        else:
            # supervised case - reward based on classification performance (enhanced metrics)
            y_pred = (self.anomaly_scores > threshold).astype(int)

            return self.calculate_reward(self.y_test, y_pred, metric=self.reward_metric)

    def calculate_reward(self, y_true, y_pred, metric='f1'):
        """Enhanced reward function with multiple metrics for anomaly detection"""
        from sklearn.metrics import (f1_score, precision_score, recall_score,
                                     accuracy_score, balanced_accuracy_score,
                                     confusion_matrix, roc_auc_score)

        try:
            if metric == 'f1':
                return f1_score(y_true, y_pred)
            elif metric == 'precision':
                return precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                return recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'balanced_accuracy':
                return balanced_accuracy_score(y_true, y_pred)
            elif metric == 'auc':
                # for binary classification only
                if len(np.unique(y_pred)) > 1:
                    return roc_auc_score(y_true, y_pred)
                return 0.5 # default AUC when no variation
            elif metric == 'weighted':
                # custom weighted metric balancing false positives and negatives
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                total = tn + fp + fn + tp

                # configure costs (adjustable)
                fp_cost = 0.6  # cost of false positives
                fn_cost = 0.4  # cost of false negatives

                # calculate weighted score (higher is better)
                fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

                return 1 - (fp_cost * fp_rate + fn_cost * fn_rate)

            else:
                return accuracy_score(y_true, y_pred)
        except:
            return 0.0 # fallback for edge cases

    def set_anomaly_scores(self, scores):
        """Set anomaly scores from external model"""
        self.anomaly_scores = scores

    def dynamic_threshold_adjustment(self):
        """Dynamically adjust threshold based on performance history"""
        if not hasattr(self, 'reward_history'):
            self.reward_history = []

        # only adjusting after collecting enough history
        if len(self.reward_history) > 10:
            # finding best performing threshold
            best_idx = np.argmax(self.reward_history[-10:])
            idx = self.current_threshold_idx - 5 + best_idx

            if idx < 0:
                idx = 0
            elif idx >= len(self.thresholds):
                idx = len(self.thresholds) - 1

            best_threshold = float(self.thresholds[idx])

            # new threshold centered around best performing one
            current_range = float(self.thresholds[-1] - self.thresholds[0])
            new_min = max(self.threshold_range[0], best_threshold - current_range / 4)
            new_max = min(self.threshold_range[1], best_threshold + current_range / 4)
            self.thresholds = np.linspace(new_min, new_max, len(self.thresholds))

            # reset current index to middle
            self.current_threshold_idx = len(self.thresholds) // 2

    def normalize_state(self, state):
        """Normalize state features for better learning"""
        # extract components
        threshold = state[0]
        mean_score = state[1]
        std_score = state[2]
        ratio_above = state[3]

        # normalize threshold to [0,1]
        norm_threshold = (threshold - self.threshold_range[0]) / (self.threshold_range[1] - self.threshold_range[0])

        # normalize other metrics (assume scores are between 0-1 already)
        # for std_score, normalize using a reasonable max value
        norm_std = min(std_score / 0.5, 1.0)

        return np.array([norm_threshold, mean_score, norm_std, ratio_above])
