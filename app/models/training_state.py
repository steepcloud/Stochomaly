import reflex as rx
from typing import List, Dict, Any, Optional


class TrainingState(rx.State):
    """State management for model training processes."""

    # Common training variables
    is_training: bool = False
    training_progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0

    # Training metrics
    loss_history: List[float] = []
    val_loss_history: List[float] = []

    # Neural network specific
    nn_learning_rate: float = 0.001
    nn_batch_size: int = 32
    nn_epochs: int = 20

    # RL specific
    rl_current_episode: int = 0
    rl_total_episodes: int = 0
    rl_rewards_history: List[float] = []
    rl_epsilon_history: List[float] = []
    rl_avg_q_values: List[float] = []
    rl_best_threshold: Optional[float] = None

    # Performance metrics
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    def update_progress(self, progress: float) -> None:
        """Update the training progress."""
        self.training_progress = progress

    def update_epoch(self, epoch: int) -> None:
        """Update the current epoch."""
        self.current_epoch = epoch

    def update_rl_episode(self, episode: int) -> None:
        """Update the current RL episode."""
        self.rl_current_episode = episode
        self.training_progress = episode / self.rl_total_episodes if self.rl_total_episodes > 0 else 0

    def start_training(self, total_epochs: int = None, total_episodes: int = None) -> None:
        """Start the training process."""
        self.is_training = True
        if total_epochs is not None:
            self.total_epochs = total_epochs
        if total_episodes is not None:
            self.rl_total_episodes = total_episodes
        self.training_progress = 0.0

    def end_training(self) -> None:
        """End the training process."""
        self.is_training = False
        self.training_progress = 1.0
        return rx.toast("Training completed successfully!", status="success")

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics."""
        if "train_accuracy" in metrics:
            self.train_accuracy = metrics["train_accuracy"]
        if "val_accuracy" in metrics:
            self.val_accuracy = metrics["val_accuracy"]
        if "precision" in metrics:
            self.precision = metrics["precision"]
        if "recall" in metrics:
            self.recall = metrics["recall"]
        if "f1_score" in metrics:
            self.f1_score = metrics["f1_score"]

    def add_loss(self, train_loss: float, val_loss: Optional[float] = None) -> None:
        """Add a loss value to the history."""
        self.loss_history.append(train_loss)
        if val_loss is not None:
            self.val_loss_history.append(val_loss)

    def add_rl_data(self, reward: float, epsilon: float = None, avg_q: float = None) -> None:
        """Add RL training data point."""
        self.rl_rewards_history.append(reward)
        if epsilon is not None:
            self.rl_epsilon_history.append(epsilon)
        if avg_q is not None:
            self.rl_avg_q_values.append(avg_q)

    def set_best_threshold(self, threshold: float) -> None:
        """Set the best threshold found by RL."""
        self.rl_best_threshold = threshold
        return rx.toast(f"Best threshold: {threshold:.4f}", status="success")

    def reset_training_state(self) -> None:
        """Reset the training state."""
        self.is_training = False
        self.training_progress = 0.0
        self.current_epoch = 0
        self.rl_current_episode = 0
        self.loss_history = []
        self.val_loss_history = []
        self.rl_rewards_history = []
        self.rl_epsilon_history = []
        self.rl_avg_q_values = []
        return rx.toast("Training state reset", status="info")