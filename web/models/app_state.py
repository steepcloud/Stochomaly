import reflex as rx
import numpy as np
from enum import Enum
from cli import create_agent
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import load_data, preprocess_data
from feature_engineering.pca import PCA
from feature_engineering.autoencoder import Autoencoder
from feature_engineering.manifold import UMAP
from feature_engineering.pipeline import FeatureEngineeringPipeline
from trainer.train import Trainer


class AppState(rx.State):
    current_step: int = 1  # 1=Data, 2=Features, 3=Training, 4=Evaluation

    # data parameters
    data_source: str = "xor"
    dataset_name: str = "iris"
    csv_filepath: str = ""
    target_column: str = ""
    scaler_type: str = "minmax"

    # data storage
    X_train: list = []
    X_test: list = []
    y_train: list = []
    y_test: list = []

    # feature engineering
    feature_method: str = "none"
    output_dim: int = 2
    ae_hidden_dim: int = 8
    ae_epochs: int = 50
    ae_batch_size: int = 32
    n_neighbors: int = 15
    min_dist: float = 0.1
    n_iter: int = 200

    # Model parameters
    model_mode: str = "nn"  # nn or rl

    # RL parameters
    rl_agent_type: str = "dqn" # dqn, double_dqn, dueling_dqn, a2c
    rl_policy: str = "epsilon-greedy" # epsilon-greedy, softmax
    rl_reward_metric: str = "f1"
    rl_threshold_min: float = 0.0
    rl_threshold_max: float = 1.0
    rl_n_thresholds: int = 10
    rl_epsilon: float = 1.0
    rl_epsilon_min: float = 0.01
    rl_epsilon_decay: float = 0.99
    rl_temperature: float = 1.0
    rl_temperature_min: float = 0.1
    rl_temperature_decay: float = 0.95
    rl_learning_rate: float = 0.001
    rl_gamma: float = 0.99  # discount factor
    rl_batch_size: int = 32
    rl_memory_size: int = 10000
    rl_target_update: int = 10  # update target network every n episodes
    rl_episodes: int = 100
    rl_max_steps: int = 100
    enable_dynamic_threshold: bool = False
    adjustment_frequency: int = 10

    # NN parameters
    epochs: int = 1000
    activation: str = "gelu"
    output_activation: str = "sigmoid"
    optimizer: str = "adam"
    hidden_size: int = 4
    learning_rate: float = 0.01
    batch_size: int = 32
    model_path: str = ""
    use_batch_norm: bool = False
    use_bayesian: bool = False

    # training state
    is_training: bool = False
    training_progress: float = 0
    loss_history: list = []

    # evaluation results
    train_accuracy: float = 0
    test_accuracy: float = 0
    precision: float = 0
    recall: float = 0
    f1_score: float = 0

    def load_data(self):
        """Load data based on selected source"""
        try:
            data_params = {}
            if self.data_source == 'sklearn':
                data_params['dataset_name'] = self.dataset_name
            elif self.data_source == 'csv':
                data_params['filepath'] = self.csv_filepath
                data_params['target_col'] = self.target_column

            X, y = load_data(source=self.data_source, **data_params)

            # binary conversion for multi-class
            if self.data_source == "sklearn" and np.max(y) > 1:
                y = (y > 0).astype(int)

            X_train, X_test, y_train, y_test = preprocess_data(X, y, scaler_type=self.scaler_type)

            # store as lists for state management
            self.X_train = X_train.tolist()
            self.X_test = X_test.tolist()
            self.y_train = y_train.tolist()
            self.y_test = y_test.tolist()

            self.current_step = 2  # move to feature engineering step
            return rx.toast("Data loaded successfully!", duration=3000)
        except Exception as e:
            return rx.toast(f"Error loading data: {str(e)}", duration=5000, color_scheme="red")

    def set_data_source(self, data_source: str):
        self.data_source = data_source

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name

    def set_target_column(self, target_column: str):
        self.target_column = target_column

    def set_scaler_type(self, scaler_type: str):
        self.scaler_type = scaler_type

    def set_feature_method(self, method: str):
        self.feature_method = method

    def set_output_dim(self, dim: int):
        self.output_dim = dim

    def set_rl_agent_type(self, agent_type: str):
        self.rl_agent_type = agent_type

    def set_rl_policy(self, policy: str):
        self.rl_policy = policy

    def set_rl_reward_metric(self, metric: str):
        self.rl_reward_metric = metric

    def set_rl_threshold_min(self, value: float):
        self.rl_threshold_min = value

    def set_rl_threshold_max(self, value: float):
        self.rl_threshold_max = value

    def set_rl_n_thresholds(self, value: int):
        self.rl_n_thresholds = value

    def set_rl_epsilon(self, value: float):
        self.rl_epsilon = value

    def set_rl_epsilon_min(self, value: float):
        self.rl_epsilon_min = value

    def set_rl_epsilon_decay(self, value: float):
        self.rl_epsilon_decay = value

    def set_rl_temperature(self, value: float):
        self.rl_temperature = value

    def set_rl_temperature_min(self, value: float):
        self.rl_temperature_min = value

    def set_rl_temperature_decay(self, value: float):
        self.rl_temperature_decay = value

    def set_rl_learning_rate(self, value: float):
        self.rl_learning_rate = value

    def set_rl_gamma(self, value: float):
        self.rl_gamma = value

    def set_rl_batch_size(self, value: int):
        self.rl_batch_size = value

    def set_rl_memory_size(self, value: int):
        self.rl_memory_size = value

    def set_rl_target_update(self, value: int):
        self.rl_target_update = value

    def set_rl_episodes(self, value: int):
        self.rl_episodes = value

    def set_rl_max_steps(self, value: int):
        self.rl_max_steps = value

    def set_enable_dynamic_threshold(self, value: bool):
        self.enable_dynamic_threshold = value

    def set_adjustment_frequency(self, value: int):
        self.adjustment_frequency = value

    def set_ae_hidden_dim(self, dim: int):
        self.ae_hidden_dim = dim

    def set_ae_epochs(self, epochs: int):
        self.ae_epochs = epochs

    def set_ae_batch_size(self, batch_size: int):
        self.ae_batch_size = batch_size

    def set_n_neighbors(self, n: int):
        self.n_neighbors = n

    def set_min_dist(self, dist: float):
        self.min_dist = dist

    def set_n_iter(self, n: int):
        self.n_iter = n

    def set_model_mode(self, mode: str):
        self.model_mode = mode

    def set_epochs(self, epochs: int):
        self.epochs = epochs

    def set_activation(self, activation: str):
        self.activation = activation

    def set_output_activation(self, activation: str):
        self.output_activation = activation

    def set_optimizer(self, optimizer: str):
        self.optimizer = optimizer

    def set_hidden_size(self, size: int):
        self.hidden_size = size

    def set_learning_rate(self, lr: float):
        self.learning_rate = lr

    def set_batch_size(self, size: int):
        self.batch_size = size

    def set_use_batch_norm(self, value: bool):
        self.use_batch_norm = value

    def set_use_bayesian(self, value: bool):
        self.use_bayesian = value

    def handle_upload(self, files: list):
        """Handle file upload for CSV data"""
        if not files or len(files) == 0:
            return rx.toast("No file uploaded", color_scheme="red")

        file = files[0]
        self.csv_filepath = file["path"]
        return rx.toast(f"File uploaded: {file['name']}")

    def apply_feature_engineering(self):
        """Apply selected feature engineering method"""
        try:
            if len(self.X_train) == 0:
                return rx.toast("No data loaded. Please load data first.", color_scheme="red")

            X_train = np.array(self.X_train)
            X_test = np.array(self.X_test)

            # skip if no feature engineering is selected
            if self.feature_method == "none":
                self.current_step = 3  # move to training step
                return rx.toast("Proceeding with original features")

            if self.feature_method == "pca":
                transformer = PCA(n_components=self.output_dim)

            elif self.feature_method == "autoencoder":
                transformer = Autoencoder(
                    input_dim=X_train.shape[1],
                    hidden_dim=self.ae_hidden_dim,
                    latent_dim=self.output_dim,
                    epochs=self.ae_epochs,
                    batch_size=self.ae_batch_size
                )

            elif self.feature_method == "umap":
                n_neighbors = self.n_neighbors
                if len(X_train) <= n_neighbors:
                    n_neighbors = max(2, len(X_train) - 1)

                transformer = UMAP(
                    n_components=self.output_dim,
                    n_neighbors=n_neighbors,
                    min_dist=self.min_dist,
                    n_iter=self.n_iter
                )

            elif self.feature_method == "enhanced":
                transformer = FeatureEngineeringPipeline(
                    autoencoder_params={
                        'input_dim': X_train.shape[1],
                        'hidden_dim': self.ae_hidden_dim,
                        'latent_dim': self.output_dim,
                        'epochs': self.ae_epochs,
                        'batch_size': self.ae_batch_size
                    },
                    if_params={
                        'n_estimators': 100,
                        'contamination': 0.1
                    },
                    lof_params={
                        'n_neighbors': 20
                    }
                )

            X_train = transformer.fit_transform(X_train)
            X_test = transformer.transform(X_test)

            self.X_train = X_train.tolist()
            self.X_test = X_test.tolist()

            self.current_step = 3  # move to training step
            return rx.toast(f"Applied {self.feature_method} feature engineering", duration=3000)

        except Exception as e:
            return rx.toast(f"Feature engineering error: {str(e)}", duration=5000, color_scheme="red")

    @rx.event(background=True)
    async def start_training(self):
        """Start the model training process in background"""
        async with self:
            self.is_training = True
        await self._train_model()

    async def _train_model(self):
        """Background task for model training"""
        try:
            X_train = np.array(self.X_train)
            y_train = np.array(self.y_train)
            X_test = np.array(self.X_test)
            y_test = np.array(self.y_test)

            # split for validation
            val_size = max(1, int(0.2 * len(X_train)))
            X_val, y_val = X_train[:val_size], y_train[:val_size]
            X_train, y_train = X_train[val_size:], y_train[val_size:]

            # initialize trainer
            trainer = Trainer(
                input_size=X_train.shape[1],
                hidden_size=self.hidden_size,
                output_size=1,
                activation=self.activation,
                output_activation=self.output_activation,
                optimizer=self.optimizer,
                learning_rate=self.learning_rate,
                use_batch_norm=self.use_batch_norm,
                use_bayesian=self.use_bayesian
            )

            # define callback to update progress
            def progress_callback(epoch, epochs, loss):
                self.training_progress = (epoch + 1) / epochs
                if len(self.loss_history) < epochs:
                    self.loss_history.append(float(loss))
                else:
                    self.loss_history[epoch] = float(loss)

            # train model
            loss_history = trainer.train(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                epochs=self.epochs,
                batch_size=self.batch_size,
                progress_callback=progress_callback
            )

            # save model if path is provided
            if self.model_path:
                trainer.save_model(self.model_path)

            # evaluate model
            self._evaluate_model(trainer, X_train, y_train, X_test, y_test)

            self.is_training = False
            self.current_step = 4  # move to evaluation step
            return rx.toast("Training completed successfully!", duration=3000)

        except Exception as e:
            self.is_training = False
            return rx.toast(f"Training error: {str(e)}", duration=5000, color_scheme="red")

    def _evaluate_model(self, trainer, X_train, y_train, X_test, y_test):
        """Evaluate the trained model and store metrics"""
        from sklearn.metrics import precision_score, recall_score, f1_score

        # get predictions
        train_preds = trainer.predict(X_train)
        test_preds = trainer.predict(X_test)

        # convert to binary predictions
        train_binary = np.round(train_preds).astype(int)
        test_binary = np.round(test_preds).astype(int)

        # calculate metrics
        self.train_accuracy = float(np.mean(train_binary == y_train))
        self.test_accuracy = float(np.mean(test_binary == y_test))
        self.precision = float(precision_score(y_test, test_binary, zero_division=0))
        self.recall = float(recall_score(y_test, test_binary, zero_division=0))
        self.f1_score = float(f1_score(y_test, test_binary, zero_division=0))

    @rx.event(background=True)
    async def train_rl_agent(self):
        """Train reinforcement learning agent for anomaly detection"""
        async with self:
            self.is_training = True
        await self._train_rl_model()

    async def _train_rl_model(self):
        """Background task for RL training"""
        try:
            from reinforcement.environment import AnomalyDetectionEnv
            from reinforcement.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, A2CAgent
            from reinforcement.policies import EpsilonGreedyPolicy, SoftmaxPolicy
            from reinforcement.training import train_rl_agent, evaluate_rl_agent

            X_train = np.array(self.X_train)
            y_train = np.array(self.y_train)
            X_test = np.array(self.X_test)
            y_test = np.array(self.y_test)

            # create environment
            env = AnomalyDetectionEnv(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                reward_metric=self.rl_reward_metric,
                threshold_range=(self.rl_threshold_min, self.rl_threshold_max),
                n_thresholds=self.rl_n_thresholds,
                use_dynamic_thresholds=self.enable_dynamic_thresholds,
                adjustment_frequency=self.adjustment_frequency
            )

            if self.rl_policy == "epsilon-greedy":
                policy = EpsilonGreedyPolicy(
                    epsilon_start=self.rl_epsilon,
                    epsilon_end=self.rl_epsilon_min,
                    epsilon_decay=self.rl_epsilon_decay
                )
            else: # softmax
                policy = SoftmaxPolicy(
                    temperature=self.rl_temperature,
                    temperature_decay=self.rl_temperature_decay,
                    temperature_min=self.rl_temperature_min
                )

            agent_params = {
                'learning_rate': self.rl_learning_rate,
                'discount_factor': self.rl_gamma,
                'batch_size': self.rl_batch_size,
                'policy': policy,
                'epsilon_start': self.rl_epsilon,
                'epsilon_end': self.rl_epsilon_min,
                'epsilon_decay': self.rl_epsilon_decay,
                'memory_size': self.rl_memory_size,
                'update_target_every': self.rl_target_update
            }

            agent = create_agent(
                self.rl_agent_type,
                state_size=4,
                action_size=3,
                **agent_params
            )

            # training function with progress callback
            def rl_progress_callback(episode, total_episodes, reward):
                self.training_progress = (episode + 1) / total_episodes
                if len(self.loss_history) < total_episodes:
                    self.loss_history.append(-float(reward))  # negative reward as "loss"
                else:
                    self.loss_history[episode] = -float(reward)

            # train agent
            results = train_rl_agent(
                agent=agent,
                environment=env,
                episodes=self.rl_episodes,
                max_steps=self.rl_max_steps,
                verbose=1,
                progress_callback=rl_progress_callback
            )

            # evaluate the agent
            eval_reward = evaluate_rl_agent(agent=agent, environment=env, episodes=10)

            # get final threshold
            env.reset()
            done = False
            while not done:
                action = agent.get_action(env.get_state())
                _, _, done, _ = env.step(action)

            final_threshold = env.thresholds[env.current_threshold_idx]

            # calculate metrics with the final threshold
            y_pred = np.array(env.anomaly_scores > final_threshold, dtype=int)

            # store metrics
            self.test_accuracy = float(np.mean(y_pred == y_test))

            from sklearn.metrics import precision_score, recall_score, f1_score
            self.precision = float(precision_score(y_test, y_pred, zero_division=0))
            self.recall = float(recall_score(y_test, y_pred, zero_division=0))
            self.f1_score = float(f1_score(y_test, y_pred, zero_division=0))

            self.is_training = False
            self.current_step = 4  # move to evaluation step
            return rx.toast("RL training completed!", duration=3000)

        except Exception as e:
            self.is_training = False
            return rx.toast(f"RL training error: {str(e)}", duration=5000, color_scheme="red")

    def reset(self):
        """Reset application state"""
        self.current_step = 1
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.loss_history = []
        self.training_progress = 0
        self.is_training = False
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        return rx.toast("Application state reset", duration=2000)