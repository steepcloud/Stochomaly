import numpy as np


class BaseLRScheduler:
    """Base class for all learning rate schedulers."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.initial_lr = optimizer.learning_rate
        self.current_lr = self.initial_lr
        self.epoch = 0

    def step(self):
        """Update learning rate according to schedule."""
        self.epoch += 1
        self._update_lr()
        self.optimizer.learning_rate = self.current_lr

    def _update_lr(self):
        """To be implemented by subclasses."""
        raise NotImplementedError

    def get_lr(self):
        """Return current learning rate."""
        return self.current_lr


class StepLR(BaseLRScheduler):
    """Step Learning Rate Scheduler.

    Decays the learning rate by gamma every step_size epochs.
    """

    def __init__(self, optimizer, step_size=10, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def _update_lr(self):
        """Update learning rate."""
        if self.epoch % self.step_size == 0 and self.epoch > 0:
            self.current_lr = self.current_lr * self.gamma


class ExponentialLR(BaseLRScheduler):
    """Exponential Learning Rate Scheduler.

    Decays the learning rate by gamma every epoch.
    """

    def __init__(self, optimizer, gamma=0.9):
        super().__init__(optimizer)
        self.gamma = gamma

    def _update_lr(self):
        """Update learning rate."""
        self.current_lr = self.initial_lr * (self.gamma ** self.epoch)


class ReduceLROnPlateau(BaseLRScheduler):
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates.
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, min_lr=0):
        super().__init__(optimizer)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.num_bad_epochs = 0

    def _update_lr(self):
        """Don't update on step(), use step(metrics) instead."""
        pass

    def step(self, metrics=None):
        """Update learning rate if metrics don't improve for patience epochs."""
        if metrics is None:
            self.epoch += 1
            return

        if self.mode == 'min':
            is_better = metrics < self.best - self.threshold
        else:  # mode == 'max'
            is_better = metrics > self.best + self.threshold

        if is_better:
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.optimizer.learning_rate = self.current_lr
            self.num_bad_epochs = 0

        self.epoch += 1


class CosineAnnealingLR(BaseLRScheduler):
    """Cosine Annealing Learning Rate Scheduler.

    Set the learning rate using a cosine annealing schedule.
    """

    def __init__(self, optimizer, T_max, eta_min=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def _update_lr(self):
        """Update learning rate."""
        if self.epoch >= self.T_max:
            self.current_lr = self.eta_min
        else:
            self.current_lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                              (1 + np.cos(np.pi * self.epoch / self.T_max)) / 2