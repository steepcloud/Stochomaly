import argparse
import numpy as np
from trainer.train import Trainer
from data.preprocess import load_data, preprocess_data
from feature_engineering.pca import PCA
from feature_engineering.autoencoder import Autoencoder
from feature_engineering.manifold import UMAP, TSNE
from feature_engineering.pipeline import FeatureEngineeringPipeline
from reinforcement.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, A2CAgent
from reinforcement.policies import EpsilonGreedyPolicy, SoftmaxPolicy
from reinforcement.training import train_rl_agent, evaluate_rl_agent
from reinforcement.environment import AnomalyDetectionEnv
import os


def create_agent(agent_type, state_size, action_size, **kwargs):
    if agent_type == "dqn":
        return DQNAgent(state_size, action_size, **kwargs)
    elif agent_type == "double_dqn":
        return DoubleDQNAgent(state_size, action_size, **kwargs)
    elif agent_type == "dueling_dqn":
        return DuelingDQNAgent(state_size, action_size, **kwargs)
    elif agent_type == "a2c":
        a2c_params = {
            'hidden_size': kwargs.get('hidden_size', 64),
            'actor_lr': kwargs.get('learning_rate', 0.001),
            'critic_lr': kwargs.get('learning_rate', 0.001),
            'discount_factor': kwargs.get('discount_factor', 0.99),
            'entropy_coefficient': kwargs.get('entropy_coefficient', 0.01),
            'max_grad_norm': kwargs.get('max_grad_norm', 0.5)
        }
        return A2CAgent(state_size, action_size, **a2c_params)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    parser = argparse.ArgumentParser(description="Train and test a neural network.")
    parser.add_argument("--mode", type=str, default="nn", choices=["nn", "rl"],
                        help="Mode: neural network (nn) or reinforcement learning (rl)")
    parser.add_argument("--train", action="store_true", help="Train the neural network")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function")
    parser.add_argument("--output-activation", type=str, default="sigmoid", choices=["sigmoid", "linear"],
                        help="Activation function for output layer")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer function")
    parser.add_argument("--hidden-size", type=int, default=4, help="Number of hidden neurons")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--save-model", type=str, help="Path to save trained model")
    parser.add_argument("--load-model", type=str, help="Path to load saved model")
    parser.add_argument('--scaler', type=str, default='minmax', help='Data scaling method')
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for the optimizer")
    parser.add_argument("--dropout-rate", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("--scheduler", type=str, default=None,
                        choices=["StepLR", "ExponentialLR", "ReduceLROnPlateau", "CosineAnnealingLR"],
                        help="Learning rate scheduler to use")
    parser.add_argument("--loss-function", type=str, default="mse",
                        choices=["mse", "mae", "binary_crossentropy"],
                        help="Loss function for neural network training")
    parser.add_argument("--early-stopping-patience", type=int, default=20,
                        help="Number of epochs with no improvement after which training stops")
    parser.add_argument("--early-stopping-min-improvement", type=float, default=0.001,
                        help="Minimum change to qualify as an improvement for early stopping")
    parser.add_argument("--use-batch-norm", action="store_true", help="Enable batch normalization")
    parser.add_argument("--use-bayesian", action="store_true", help="Use Bayesian neural network")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="KL divergence weight for Bayesian NN")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples for Monte Carlo approximation")
    parser.add_argument('--data_source', type=str, default='xor',
                        help='Data source: xor, sklearn, csv')
    parser.add_argument('--dataset_name', type=str, default='iris',
                        help='Name of sklearn dataset')
    parser.add_argument('--csv_filepath', type=str, help='Path to CSV file')
    parser.add_argument('--target_col', type=str, help='Target column name for CSV data')
    parser.add_argument('--feature-engineering', type=str,
                        choices=['none', 'pca', 'autoencoder', 'umap', 'tsne'],
                        default='none', help='Feature engineering method to use')
    parser.add_argument('--output-dim', type=int, default=2,
                        help='Output dimension for feature engineering')
    parser.add_argument("--rl-agent", type=str, default="dqn",
                        choices=["dqn", "double_dqn", "dueling_dqn", "a2c"],
                        help="Type of reinforcement learning agent to use")

    # Scheduler-specific parameters
    # StepLR parameters
    parser.add_argument("--step-size", type=int, default=10,
                        help="Period of learning rate decay (for StepLR)")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Multiplicative factor of learning rate decay")

    # ReduceLROnPlateau parameters
    parser.add_argument("--scheduler-mode", type=str, default="min", choices=["min", "max"],
                        help="Mode for ReduceLROnPlateau: min - reduce when metric stops decreasing, "
                             "max - reduce when metric stops increasing")
    parser.add_argument("--factor", type=float, default=0.1,
                        help="Factor by which the learning rate will be reduced (for ReduceLROnPlateau)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Number of epochs with no improvement after which LR will be reduced"
                             " (for ReduceLROnPlateau)")
    parser.add_argument("--threshold", type=float, default=1e-4,
                        help="Threshold for measuring the new optimum (for ReduceLROnPlateau)")
    parser.add_argument("--min-lr", type=float, default=0,
                        help="Minimum learning rate (for ReduceLROnPlateau)")

    # CosineAnnealingLR parameters
    parser.add_argument("--t-max", type=int, default=50,
                        help="Maximum number of iterations/epochs (for CosineAnnealingLR)")
    parser.add_argument("--eta-min", type=float, default=0,
                        help="Minimum learning rate (for CosineAnnealingLR)")

    # UMAP specific parameters
    parser.add_argument('--n-neighbors', type=int, default=15,
                        help='Number of neighbors for UMAP')
    parser.add_argument('--min-dist', type=float, default=0.1,
                        help='Minimum distance parameter for UMAP')
    parser.add_argument('--n-iter', type=int, default=200,
                        help='Number of iterations for UMAP optimization')

    # t-SNE specific parameters
    parser.add_argument('--perplexity', type=float, default=30.0,
                        help='Perplexity parameter for t-SNE (balance between local and global structure)')
    parser.add_argument('--tsne-learning-rate', type=float, default=200.0,
                        help='Learning rate for t-SNE')
    parser.add_argument('--tsne-iterations', type=int, default=1000,
                        help='Number of iterations for t-SNE optimization')

    # Autoencoder specific parameters
    parser.add_argument('--ae-hidden-dim', type=int, default=8,
                        help='Hidden layer dimension for autoencoder')
    parser.add_argument('--ae-epochs', type=int, default=50,
                        help='Number of epochs for autoencoder training')
    parser.add_argument('--ae-batch-size', type=int, default=32,
                        help='Batch size for autoencoder training')

    # Reinforcement Learning arguments
    parser.add_argument("--rl-episodes", type=int, default=100, help="Number of episodes for RL training")
    parser.add_argument("--rl-max-steps", type=int, default=100, help="Maximum steps per episode for RL")
    parser.add_argument("--rl-policy", type=str, default="epsilon-greedy", choices=["epsilon-greedy", "softmax"],
                        help="Policy for action selection")
    parser.add_argument("--rl-epsilon", type=float, default=1.0, help="Initial epsilon for epsilon-greedy policy")
    parser.add_argument("--rl-epsilon-decay", type=float, default=0.99, help="Decay rate for epsilon")
    parser.add_argument("--rl-epsilon-min", type=float, default=0.01, help="Minimum epsilon value")
    parser.add_argument("--rl-temperature", type=float, default=1.0, help="Temperature for softmax policy")
    parser.add_argument("--rl-temperature-decay", type=float, default=0.995,
                        help="Decay rate for temperature in softmax policy")
    parser.add_argument("--rl-temperature-min", type=float, default=0.1,
                        help="Minimum temperature value for softmax policy")
    parser.add_argument("--rl-gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--rl-batch-size", type=int, default=32, help="Batch size for RL training")
    parser.add_argument("--rl-learning-rate", type=float, default=0.001, help="Learning rate for RL agent")
    parser.add_argument("--rl-target-update", type=int, default=10, help="Update frequency for target network")
    parser.add_argument("--rl-memory-size", type=int, default=10000, help="Size of replay buffer")
    parser.add_argument("--rl-reward-metric", type=str, default="f1", choices=["f1", "accuracy"],
                        help="Metric to use for rewards in anomaly detection")
    parser.add_argument("--rl-n-thresholds", type=int, default=10,
                        help="Number of thresholds for anomaly detection environment")
    parser.add_argument("--rl-threshold-range", type=str, default="0.0,1.0",
                        help="Range of thresholds (min,max) for anomaly detection")

    # Environment specific parameters
    parser.add_argument("--rl-reward-metric", type=str, default="f1",
                        choices=["f1", "precision", "recall", "balanced_accuracy", "weighted", "auc"],
                        help="Metric to use for rewards in anomaly detection")
    parser.add_argument("--enable-dynamic-threshold", action="store_true",
                        help="Enable dynamic threshold adjustment during training")
    parser.add_argument("--adjustment-frequency", type=int, default=20,
                        help="How often to adjust thresholds (steps)")
    parser.add_argument("--reward-alpha", type=float, default=0.5,
                        help="Weight factor for balancing precision and recall in weighted metrics")

    # Enhanced feature engineering pipeline
    parser.add_argument("--use-enhanced-features", action="store_true",
                        help="Use enhanced feature engineering pipeline (autoencoder + isolation forest + LOF)")
    parser.add_argument("--ae-latent-dim", type=int, default=10, help="Autoencoder latent dimension")
    parser.add_argument("--if-n-estimators", type=int, default=100, help="Isolation Forest estimators")
    parser.add_argument("--if-contamination", type=float, default=0.1,
                        help="Isolation Forest expected outlier proportion")
    parser.add_argument("--lof-n-neighbors", type=int, default=20, help="LOF neighbors count")

    args = parser.parse_args()

    data_params = {}
    if args.data_source == 'sklearn':
        data_params['dataset_name'] = args.dataset_name
    elif args.data_source == 'csv':
        data_params['filepath'] = args.csv_filepath
        data_params['target_col'] = args.target_col

    # Load and preprocess data
    X, y = load_data(source=args.data_source, **data_params)

    # for sklearn datasets with multiple classes, we need binary targets for this test
    if args.data_source == "sklearn" and y.max() > 1:
        # convert to binary problem (0 vs rest)
        y = (y > 0).astype(int)

    X_train, X_test, y_train, y_test = preprocess_data(X, y, scaler_type=args.scaler)

    if args.use_enhanced_features:
        print("Applying enhanced feature engineeering pipeline...")

        pipeline = FeatureEngineeringPipeline(
            autoencoder_params = {
                'input_dim': X_train.shape[1],
                'hidden_dim': args.ae_hidden_dim,
                'latent_dim': args.ae_latent_dim,
                'epochs': args.ae_epochs,
                'batch_size': args.ae_batch_size
            },
            if_params = {
                'n_estimators': args.if_n_estimators,
                'contamination': args.if_contamination
            },
            lof_params = {
                'n_neighbors': args.lof_n_neighbors
            }
        )

        pipeline.fit(X_train)
        X_train = pipeline.transform(X_train)
        X_test = pipeline.transform(X_test)
        print(f"Enhanced features: {X_train.shape[1]} dimensions")

        # skipping regular feature engineering if enhanced is used
        args.feature_engineering = 'none'

    if args.feature_engineering != 'none':
        print(f"Applying {args.feature_engineering} feature engineering...")

        if args.feature_engineering == 'pca':
            transformer = PCA(n_components=args.output_dim)

        elif args.feature_engineering == 'autoencoder':
            transformer = Autoencoder(
                input_dim=X_train.shape[1],
                hidden_dim=args.ae_hidden_dim,
                latent_dim=args.output_dim,
                epochs=args.ae_epochs,
                batch_size=args.ae_batch_size
            )

        elif args.feature_engineering == 'umap':
            n_neighbors = args.n_neighbors
            if len(X_train) <= n_neighbors:
                n_neighbors = max(2, len(X_train) - 1)
                print(f"WARNING: Dataset too small for requested n_neighbors={args.n_neighbors}.")
                print(f"Automatically adjusting to n_neighbors={n_neighbors}")

            transformer = UMAP(
                n_components=args.output_dim,
                n_neighbors=n_neighbors,
                min_dist=args.min_dist,
                n_iter=args.n_iter
            )

        elif args.feature_engineering == 'tsne':
            transformer = TSNE(
                n_components=args.output_dim,
                perplexity=args.perplexity,
                learning_rate=args.tsne_learning_rate,
                n_iter=args.tsne_iterations
            )

        X_train = transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)
        print(f"Reduced dimension to {X_train.shape[1]} features")

    if args.mode == "nn":
        # Neural network training/prediction mode

        # Split training data into training and validation sets
        val_size = max(1, int(0.2 * len(X_train))) # 20% for validation
        X_val, y_val = X_train[:val_size], y_train[:val_size]
        X_train, y_train = X_train[val_size:], y_train[val_size:]

        scheduler_type = args.scheduler
        scheduler_params = {}

        if scheduler_type == "StepLR":
            scheduler_params = {
                "step_size": args.step_size,
                "gamma": args.gamma
            }
        elif scheduler_type == "ExponentialLR":
            scheduler_params = {
                "gamma": args.gamma  # ExponentialLR uses the same gamma parameter
            }
        elif scheduler_type == "ReduceLROnPlateau":
            scheduler_params = {
                "mode": args.scheduler_mode,
                "factor": args.factor,
                "patience": args.patience,
                "threshold": args.threshold,
                "min_lr": args.min_lr
            }
        elif scheduler_type == "CosineAnnealingLR":
            scheduler_params = {
                "T_max": args.t_max,
                "eta_min": args.eta_min
            }

        # Initialize trainer with correct input size
        trainer = Trainer(
            input_size=X_train.shape[1],
            hidden_size=args.hidden_size,
            output_size=1,
            activation=args.activation,
            output_activation=args.output_activation,
            optimizer=args.optimizer,
            learning_rate=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            dropout_rate=args.dropout_rate,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_improvement=args.early_stopping_min_improvement,
            scheduler_type=scheduler_type,
            scheduler_params=scheduler_params,
            use_batch_norm=args.use_batch_norm,
            use_bayesian=args.use_bayesian,
            kl_weight=args.kl_weight,
            loss_function=args.loss_function
        )

        # Load model if specified
        if args.load_model:
            if os.path.exists(args.load_model):
                trainer.load_model(args.load_model)
            else:
                print(f"Warning: Model file '{args.load_model}' not found. Proceeding without loading.")

        if args.train:
            # Train the model
            loss_history = trainer.train(X_train, y_train, X_val=X_val, y_val=y_val,
                                         epochs=args.epochs, batch_size=args.batch_size, n_samples=args.n_samples)

            # Check if training was stopped early
            if trainer.stopped_early:
                print("Training stopped early due to early stopping.")

            # Save model if required
            if args.save_model:
                trainer.save_model(args.save_model)

            # Evaluate on both train and test sets
            train_predictions = trainer.predict(X_train)
            test_predictions = trainer.predict(X_test)

            # Convert predictions to binary classification results
            train_predictions = np.round(train_predictions)
            test_predictions = np.round(test_predictions)

            train_accuracy = np.mean(train_predictions == y_train)
            test_accuracy = np.mean(test_predictions == y_test)

            print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
            print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        else:
            # Make predictions on both sets
            train_predictions = trainer.predict(X_train)
            test_predictions = trainer.predict(X_test)

            print("Training Set Predictions:")
            print(np.round(train_predictions))
            print("\nTest Set Predictions:")
            print(np.round(test_predictions))

    elif args.mode == "rl":
        # Reinforcement Learning for anomaly detection
        print("Running reinforcement learning for anomaly detection")

        # parse threshold range
        threshold_min, threshold_max = map(float, args.rl_threshold_range.split(','))

        # create environment
        env = AnomalyDetectionEnv(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            reward_metric=args.rl_reward_metric,
            threshold_range=(threshold_min, threshold_max),
            n_thresholds=args.rl_n_thresholds,
            use_dynamic_thresholds=args.enable_dynamic_threshold,
            adjustment_frequency=args.adjustment_frequency
        )

        # create policy
        if args.rl_policy == "epsilon-greedy":
            policy = EpsilonGreedyPolicy(
                epsilon_start=args.rl_epsilon,
                epsilon_end=args.rl_epsilon_min,
                epsilon_decay=args.rl_epsilon_decay
            )
        else:  # softmax
            policy = SoftmaxPolicy(
                temperature=args.rl_temperature,
                temperature_decay=args.rl_temperature_decay,
                temperature_min=args.rl_temperature_min
            )

        # create agent
        state_size = 4  # based on AnomalyDetectionEnv's state representation
        action_size = 3  # decrease, keep, increase threshold

        agent = create_agent(
            agent_type=args.rl_agent,
            state_size=state_size,
            action_size=action_size,
            learning_rate=args.rl_learning_rate,
            discount_factor=args.rl_gamma,
            epsilon_start=args.rl_epsilon,
            epsilon_end=args.rl_epsilon_min,
            epsilon_decay=args.rl_epsilon_decay,
            batch_size=args.rl_batch_size,
            update_target_every=args.rl_target_update,
            policy=policy,
            memory_size=args.rl_memory_size
        )

        # run training
        print(f"Training RL agent for {args.rl_episodes} episodes...")
        results = train_rl_agent(
            agent=agent,
            environment=env,
            episodes=args.rl_episodes,
            max_steps=args.rl_max_steps,
            verbose=1
        )

        # evaluate
        print("Evaluating RL agent...")
        eval_reward = evaluate_rl_agent(
            agent=agent,
            environment=env,
            episodes=10
        )
        print(f"Evaluation average reward: {eval_reward:.4f}")

        # get the final threshold selected by the agent
        env.reset()
        done = False
        while not done:
            action = agent.get_action(env.get_state())
            _, _, done, _ = env.step(action)

        final_threshold = env.thresholds[env.current_threshold_idx]
        print(f"Final selected threshold: {final_threshold:.4f}")

        # calculate anomaly predictions using the final threshold
        y_pred = np.array(env.anomaly_scores > final_threshold, dtype=int)
        accuracy = np.mean(y_pred == y_test)

        print(f"Anomaly detection accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
