import argparse
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
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
from plot_utils import *
import os
from sklearn import datasets


warnings.filterwarnings("ignore", message="A single label was found")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


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
    parser.add_argument('--scaler', type=str, default='minmax',
                        choices=['standard', 'minmax', 'robust'],
                        help='Data scaling method (standard, minmax, or robust)')
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
    X, y_loaded, feature_names_loaded = load_data(source=args.data_source, return_feature_names=True, **data_params)
    print(f"[DEBUG] Data loaded by load_data: X.shape={X.shape}, y_loaded.shape={y_loaded.shape}")

    y_original_full = np.asarray(y_loaded).flatten()
    y_for_processing = y_original_full.copy()

    class_label_names = None

    if args.data_source == "sklearn":
        if args.dataset_name == 'iris':
               class_label_names = datasets.load_iris().target_names
        elif args.dataset_name == 'wine':
            class_label_names = datasets.load_wine().target_names
        elif args.dataset_name == 'breast_cancer':
            class_label_names = datasets.load_breast_cancer().target_names
        elif args.dataset_name == 'digits':
            class_label_names = [f"Digit {i}" for i in range(10)]
        elif args.dataset_name == 'diabetes':
            class_label_names = ["Low Progression", "High Progression"]
            median_val = np.median(y_original_full)
            y_for_processing = (y_original_full > median_val).astype(int)
            y_for_plotting = y_for_processing
        if class_label_names is not None:
            print(f"[DEBUG] Loaded class label names for {args.dataset_name}: {class_label_names}")
        
        if args.dataset_name == 'diabetes':
            pass
        elif y_for_processing.ndim == 1 and np.max(y_for_processing) > 0 and len(np.unique(y_for_processing)) > 2:
            print(f"[INFO] Converting multi-class target to binary (0 vs. rest) for dataset: {args.dataset_name}")
            y_for_processing = (y_for_processing > 0).astype(int)
        
    '''
    # for sklearn datasets with multiple classes, we need binary targets for this test
    if args.data_source == "sklearn" and y.ndim == 1 and y.max() > 1:
        # convert to binary problem (0 vs rest)
        print(f"[INFO] Converting multi-class target to binary (0 vs. rest) for dataset: {args.dataset_name}")
        y = (y > 0).astype(int)
    '''

    #X_train, X_test, y_train, y_test = preprocess_data(X, y, scaler_type=args.scaler)
    
    preprocess_test_size = 0.25
    preprocess_random_state = 42

    X_train, X_test, y_train, y_test = preprocess_data(X, y_for_processing, 
                                                       scaler_type=args.scaler,
                                                       test_size=preprocess_test_size,
                                                       random_state=preprocess_random_state)

    # ensuring y_test is 1D for consistent metric calculations
    y_test = np.asarray(y_test).flatten()
    y_train = np.asarray(y_train).flatten()

    y_test_original_labels = None
    if X.shape[0] == y_original_full.shape[0]:
        try:
            stratify_target_for_original_split = y_for_processing if len(np.unique(y_for_processing)) > 1 else None

            _, _, _, y_test_original_labels_split = train_test_split(
                X,
                y_original_full,
                test_size=preprocess_test_size,
                random_state=preprocess_random_state,
                stratify=stratify_target_for_original_split
            )
            y_test_original_labels = y_test_original_labels_split

            if len(y_test_original_labels) != len(y_test):
                print(f"[WARNING] Length mismatch: y_test ({len(y_test)}) vs y_test_original_labels ({len(y_test_original_labels)}). Original labels for plotting might be incorrect.")
                y_test_original_labels = y_test.copy()
            else:
                print(f"[DEBUG] Successfully mapped y_test to original labels. Example: {y_test_original_labels[:5]}")
        except Exception as e:
            print(f"[WARNING] Could not deterministically split original y labels to match X_test due to: {e}. Using binary y_test as fallback for original labels.")
            y_test_original_labels = y_test.copy()
    else:
        print("[WARNING] Initial X and y_original_full shapes mismatch. Cannot derive y_test_original_labels. Using binary y_test as fallback.")
        y_test_original_labels = y_test.copy()

    print(f"[DEBUG] Initial load: X_train.shape={X_train.shape}, X_test.shape={X_test.shape}")
    print(f"[DEBUG] Initial load: y_train.shape={y_train.shape}, y_test.shape={y_test.shape}")
    if feature_names_loaded:
        print(f"[DEBUG] Loaded {len(feature_names_loaded)} feature names: {feature_names_loaded[:5]}...")
    else:
        print("[DEBUG] No feature names loaded initially (e.g. XOR data or CSV without header).")

    original_X_train_for_ae_scoring = None
    original_X_test_for_ae_scoring = None
    ae_reconstruction_errors_train = None
    ae_reconstruction_errors_test = None

    if args.feature_engineering == 'autoencoder':
        print("[DEBUG] Copying X_train and X_test for Autoencoder reconstruction error calculation.")
        original_X_train_for_ae_scoring = X_train.copy()
        original_X_test_for_ae_scoring = X_test.copy()

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

        transformer = None
        
        if args.feature_engineering == 'pca':
            transformer = PCA(n_components=args.output_dim)

        elif args.feature_engineering == 'autoencoder':
            if original_X_train_for_ae_scoring is None:
                print("[ERROR] original_X_train_for_ae_scoring is None. Cannot proceed with Autoencoder.")
                return
        
            transformer = Autoencoder(
                input_dim=original_X_train_for_ae_scoring.shape[1],
                hidden_dim=args.ae_hidden_dim,
                latent_dim=args.output_dim,
                epochs=args.ae_epochs,
                batch_size=args.ae_batch_size
            )

            print("[INFO] Fitting Autoencoder on (already scaled) original X_train data...")
            transformer.fit(original_X_train_for_ae_scoring)

            if hasattr(transformer, 'get_reconstruction_error'):
                print("[INFO] Calculating Autoencoder reconstruction errors...")
                ae_reconstruction_errors_train = transformer.get_reconstruction_error(original_X_train_for_ae_scoring)
                ae_reconstruction_errors_test = transformer.get_reconstruction_error(original_X_test_for_ae_scoring)
                print(f"[DEBUG] AE reconstruction errors generated. Train shape: {ae_reconstruction_errors_train.shape if ae_reconstruction_errors_train is not None else 'None'}, Test shape: {ae_reconstruction_errors_test.shape if ae_reconstruction_errors_test is not None else 'None'}")
            else:
                print("[WARNING] Autoencoder class is missing get_reconstruction_error method.")
                
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
        
        if transformer is not None: 
            if args.feature_engineering == 'autoencoder':
                print("[INFO] Transforming (already scaled) original X_train/X_test to latent space using Autoencoder.")
                if original_X_train_for_ae_scoring is not None:
                    X_train = transformer.transform(original_X_train_for_ae_scoring)
                if original_X_test_for_ae_scoring is not None:
                    X_test = transformer.transform(original_X_test_for_ae_scoring)
            elif args.feature_engineering in ['pca', 'umap', 'tsne']:
                # For PCA, UMAP, TSNE: fit_transform on X_train, then transform X_test.
                print(f"[INFO] Applying {args.feature_engineering}: fit_transform on X_train, transform on X_test.")
                X_train = transformer.fit_transform(X_train)
                X_test = transformer.transform(X_test)
            
            print(f"Data transformed by {args.feature_engineering}. New X_train shape: {X_train.shape}, New X_test shape: {X_test.shape}")

        #X_train = transformer.fit_transform(X_train)
        #X_test = transformer.transform(X_test)
        #print(f"Reduced dimension to {X_train.shape[1]} features")

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
            test_predictions_binary = (test_predictions > 0.5).astype(int)

            # convert predictions to binary for evaluation
            #y_train_pred_binary = (train_predictions > 0.5).astype(int)
            #y_test_pred_binary = (test_predictions > 0.5).astype(int)

            #print("\nGenerating visualizations...")

            # confusion matrix
            #plot_confusion_matrix(y_test, y_test_pred_binary, save_path="plots/confusion_matrix.png")

            # ROC curve
            #plot_roc_curve(y_test, test_predictions, save_path='plots/roc_curve.png')
            
            # prediction distribution
            #plot_prediction_distribution(test_predictions, y_test, save_path='plots/prediction_dist.png')
            
            # threshold sensitivity analysis
            #plot_threshold_sensitivity(y_test, test_predictions, save_path='plots/threshold_sensitivity.png')
            
        dataset_info_nn = args.dataset_name
        feature_names_nn = feature_names_loaded

        true_labels_for_nn_plot = y_test_original_labels if y_test_original_labels is not None and len(y_test_original_labels) == len(y_test) else y_test

        plot_confusion_matrix(y_test, test_predictions_binary, save_path="plots/nn_confusion_matrix.png")
        plot_roc_curve(y_test, test_predictions, save_path='plots/nn_roc_curve.png')
        plot_prediction_distribution(test_predictions, y_test, save_path='plots/nn_prediction_dist.png')
        plot_threshold_sensitivity(y_test, test_predictions, save_path='plots/nn_threshold_sensitivity.png')

        plot_anomaly_scatter(X_test, true_labels_for_nn_plot, test_predictions_binary, 
                           feature_names=feature_names_nn, dataset_name=dataset_info_nn,
                           class_label_names=class_label_names, # Pass class names
                           save_path='plots/nn_anomaly_scatter.png')
        plot_feature_importance(X_test, test_predictions_binary,
                              feature_names=feature_names_nn, dataset_name=dataset_info_nn,
                              save_path='plots/nn_feature_importance.png')
        plot_anomaly_distribution(X_test, test_predictions_binary,
                                feature_names=feature_names_nn, dataset_name=dataset_info_nn,
                                y_true_original=true_labels_for_nn_plot, # Pass original true labels
                                class_label_names=class_label_names,  # Pass class names
                                save_path='plots/nn_anomaly_distribution.png')

        # statistics about anomalies
        anomaly_count = test_predictions_binary.sum()
        normal_count = len(test_predictions_binary) - anomaly_count
        print(f"\nAnomaly Detection Statistics for {dataset_info_nn} dataset:")
        print(f"Total samples: {len(test_predictions_binary)}")
        print(f"Detected anomalies: {anomaly_count} ({anomaly_count/len(test_predictions_binary)*100:.1f}%)")
        print(f"Normal samples: {normal_count} ({normal_count/len(test_predictions_binary)*100:.1f}%)")

        # for numeric features, show average values in normal vs anomaly classes
        if X_test.shape[1] <= 10:  # only show if we don't have too many features
            print("\nFeature statistics (Normal vs Anomaly):")
            for i in range(X_test.shape[1]):
                feature_name = feature_names_nn[i] if feature_names_nn else f"Feature {i}"
                    
                normal_rows = np.where(test_predictions_binary == 0)[0]
                anomaly_rows = np.where(test_predictions_binary == 1)[0]
                    
                normal_mean = X_test[normal_rows, i].mean() if len(normal_rows) > 0 else 0
                anomaly_mean = X_test[anomaly_rows, i].mean() if len(anomaly_rows) > 0 else 0
                diff = abs(normal_mean - anomaly_mean)
                print(f"{feature_name}: Normal={normal_mean:.2f}, Anomaly={anomaly_mean:.2f}, Diff={diff:.2f}")
            
        print("\nVisualization complete! Check the 'plots' directory for results.")

    elif args.mode == "rl":
        # Reinforcement Learning for anomaly detection
        print("Running reinforcement learning for anomaly detection")

        # parse threshold range
        threshold_min, threshold_max = map(float, args.rl_threshold_range.split(','))

        #print(f"[DEBUG] RL Mode: X_train for env initialization:\n{X_train}")
        #print(f"[DEBUG] RL Mode: y_train for env initialization:\n{y_train}")

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
            adjustment_frequency=args.adjustment_frequency,
            max_steps=args.rl_max_steps
        )

        scores_for_rl_training = None

        if X_train.shape[0] == 0:
            print("[ERROR] RL Mode: X_train is empty. Cannot set anomaly scores or proceed with RL training.")
            return
        
        if args.feature_engineering == 'autoencoder' and ae_reconstruction_errors_train is not None:
            print("[INFO] RL Mode: Using Autoencoder reconstruction errors for training scores.")
            scores_for_rl_training = ae_reconstruction_errors_train
        elif X_train.ndim == 1:
            print(f"[DEBUG] RL Mode: Using current 1D X_train (shape: {X_train.shape}) directly as anomaly scores for training. (FE: '{args.feature_engineering}')")
            scores_for_rl_training = X_train
        elif X_train.ndim == 2 and X_train.shape[1] > 0:
            print(f"[DEBUG] RL Mode: Using first feature of current 2D X_train (X_train[:, 0], shape: {X_train.shape}) as anomaly scores for training. (FE: '{args.feature_engineering}')")
            scores_for_rl_training = X_train[:, 0]
        elif X_train.ndim == 2 and X_train.shape[1] == 0:
            print(f"[ERROR] RL Mode: X_train has samples but no features (shape: {X_train.shape}). Using zeros as dummy scores.")
            scores_for_rl_training = np.zeros(X_train.shape[0])
        else:
            print(f"[ERROR] RL Mode: X_train has an unexpected shape ({X_train.shape}). Cannot derive anomaly scores. Using zeros as dummy.")
            scores_for_rl_training = np.zeros(X_train.shape[0]) if hasattr(X_train, 'shape') and len(X_train.shape) > 0 and X_train.shape[0] > 0 else np.array([])

        if scores_for_rl_training is not None and len(scores_for_rl_training) > 0:
            env.set_anomaly_scores(scores_for_rl_training.flatten())
            print(f"[DEBUG] RL Mode: Anomaly scores (length: {len(scores_for_rl_training)}) set in environment for training.")
        else:
            print("[CRITICAL ERROR] RL Mode: Failed to derive any anomaly scores for training. Environment will fail.")
            return
            
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

        test_sample_scores = np.array([]) # Initialize

        if args.feature_engineering == 'autoencoder' and ae_reconstruction_errors_test is not None:
            print("[INFO] RL Mode: Using Autoencoder reconstruction errors for X_test scores.")
            test_sample_scores = ae_reconstruction_errors_test
        elif hasattr(env, 'score_model') and env.score_model is not None and hasattr(env.score_model, 'get_anomaly_scores'):
            print("[DEBUG] RL Mode: Using env.score_model.get_anomaly_scores() for X_test.")
            test_sample_scores = env.score_model.get_anomaly_scores(X_test)
        elif hasattr(env, 'get_anomaly_scores_for_data'): 
            print("[DEBUG] RL Mode: Using env.get_anomaly_scores_for_data(X_test).")
            test_sample_scores = env.get_anomaly_scores_for_data(X_test)
        elif X_test.ndim == 2 and X_test.shape[1] > 0 : # Check if X_test is 2D and has columns
            # Fallback: If env scores based on the first feature of X_test
            print("[DEBUG] RL Mode: No specific score model/method in env. Assuming scores are based on X_test[:, 0].")
            test_sample_scores = X_test[:, 0]
        else:
            print(f"[ERROR] RL Mode: X_test is not 2D, has no features, or no way to get scores. X_test.shape: {X_test.shape}. Cannot generate y_pred.")
            # test_sample_scores remains empty
        
        if len(test_sample_scores) > 0:
            y_pred = (test_sample_scores > final_threshold).astype(int)
        else:
            y_pred = np.array([]) # Empty predictions if no scores

        y_pred = y_pred.flatten() # Ensure y_pred is 1D

        print(f"[DEBUG] RL Mode: y_pred derived from test_sample_scores. Shape: {y_pred.shape}")
        
        # Calculate accuracy based on these per-sample predictions
        if len(y_pred) == len(y_test) and len(y_test) > 0:
            accuracy = np.mean(y_pred == y_test) # y_test is already flattened
            print(f"Anomaly detection accuracy on X_test: {accuracy * 100:.2f}%")
        elif len(y_test) == 0:
            print("Anomaly detection accuracy: N/A (empty y_test)")
        else:
            print(f"Anomaly detection accuracy: N/A (y_pred length {len(y_pred)} != y_test length {len(y_test)})")

        # Correctly set feature_names and dataset_info for RL mode plots
        #feature_names = feature_names_loaded # Use names from initial data load
        #dataset_info = args.dataset_name # Default to dataset_name argument

        dataset_info_rl = args.dataset_name
        feature_names_rl = feature_names_loaded

        # calculate anomaly predictions using the final threshold
        #y_pred = np.array(env.anomaly_scores > final_threshold, dtype=int)
        #accuracy = np.mean(y_pred == y_test)

        #print(f"Anomaly detection accuracy: {accuracy * 100:.2f}%")

        #print(f"[DEBUG] RL Mode: X_test.shape before deriving y_pred = {X_test.shape}")
        #print(f"[DEBUG] RL Mode: final_threshold = {final_threshold}")

        #dataset_info = ""
        #feature_names = None

        if feature_names_rl and X_test.ndim == 2 and X_test.shape[1] != len(feature_names_rl):
            print(f"[DEBUG] Feature names from load_data ({len(feature_names_rl)}) don't match X_test columns ({X_test.shape[1]}) after FE. Using generic names for plotting.")
            feature_names_rl = [f"Feature {i}" for i in range(X_test.shape[1])]
        elif not feature_names_rl and X_test.ndim == 2 and X_test.shape[1] > 0:
             feature_names_rl = [f"Feature {i}" for i in range(X_test.shape[1])]
        

        print(f"[DEBUG] RL Mode: Plotting with dataset_info='{dataset_info_rl}'")
        if feature_names_rl:
            print(f"[DEBUG] RL Mode: Plotting with {len(feature_names_rl)} feature names: {feature_names_rl[:5]}...")
        else:
            print("[DEBUG] RL Mode: Plotting without specific feature names.")

        print("\nGenerating detailed anomaly visualizations...")
        if len(y_pred) > 0 and len(y_pred) == X_test.shape[0]:
            if args.dataset_name == 'diabetes' and 'y_for_plotting' in locals():
                true_labels_for_rl_plot = y_for_plotting[-len(y_test):]
            else:
                true_labels_for_rl_plot = y_test_original_labels if y_test_original_labels is not None and len(y_test_original_labels) == len(y_test) else y_test

            plot_anomaly_scatter(X_test, true_labels_for_rl_plot, y_pred, 
                            feature_names=feature_names_rl, 
                            dataset_name=dataset_info_rl,
                            class_label_names=class_label_names,
                            save_path='plots/rl_anomaly_scatter.png')

            plot_feature_importance(X_test, y_pred,
                                feature_names=feature_names_rl, 
                                dataset_name=dataset_info_rl,
                                save_path='plots/rl_feature_importance.png')

            plot_anomaly_distribution(X_test, y_pred,
                                    feature_names=feature_names_rl, 
                                    dataset_name=dataset_info_rl,
                                    y_true_original=true_labels_for_rl_plot,
                                    class_label_names=class_label_names,
                                    save_path='plots/rl_anomaly_distribution.png')
            
            if len(test_sample_scores) == len(y_test):
                plot_roc_curve(y_test, test_sample_scores, save_path='plots/rl_roc_curve.png', dataset_name=dataset_info_rl)
        else:
            print("[INFO] Skipping RL detailed visualizations as y_pred is not valid for X_test.")
            print(f"[INFO] y_pred length: {len(y_pred)}, X_test samples: {X_test.shape[0]}")

        if len(y_pred) > 0 and len(y_pred) == X_test.shape[0]:
            anomaly_count = y_pred.sum()
            normal_count = len(y_pred) - anomaly_count
            print(f"\nRL Anomaly Detection Statistics for {dataset_info_rl} dataset:")
            print(f"Total samples in X_test: {X_test.shape[0]}")
            print(f"Total predictions made: {len(y_pred)}")
            print(f"Detected anomalies: {anomaly_count} ({anomaly_count/len(y_pred)*100:.1f}% of predictions)")
            print(f"Normal samples: {normal_count} ({normal_count/len(y_pred)*100:.1f}% of predictions)")

            # For numeric features, show average values in normal vs anomaly classes
            if X_test.ndim == 2 and X_test.shape[1] > 0 and X_test.shape[1] <= 10: # Check X_test is 2D and has features
                print("\nFeature statistics (Normal vs Anomaly based on RL predictions):")
                for i in range(X_test.shape[1]):
                    current_feature_name = feature_names_rl[i] if feature_names_rl and i < len(feature_names_rl) else f"Feature {i}"
                    
                    normal_rows = np.where(y_pred == 0)[0]
                    anomaly_rows = np.where(y_pred == 1)[0]
            
                    normal_mean = X_test[normal_rows, i].mean() if len(normal_rows) > 0 else np.nan
                    anomaly_mean = X_test[anomaly_rows, i].mean() if len(anomaly_rows) > 0 else np.nan
                    
                    if not (np.isnan(normal_mean) or np.isnan(anomaly_mean)):
                        diff = abs(normal_mean - anomaly_mean)
                        print(f"{current_feature_name}: Normal={normal_mean:.2f}, Anomaly={anomaly_mean:.2f}, Diff={diff:.2f}")
                    else:
                        print(f"{current_feature_name}: Normal={normal_mean:.2f}, Anomaly={anomaly_mean:.2f} (One class may be empty or no samples)")
        else:
            print("\nRL Anomaly Detection Statistics: N/A (y_pred is not valid for X_test)")
        
        # Print Anomalous Sample Details
        print("\n--- Detected Anomalous Samples (RL Mode) ---")
        if len(y_pred) > 0 and len(y_pred) == X_test.shape[0]: # Check y_pred validity
            anomaly_indices_in_test = np.where(y_pred == 1)[0]
            if len(anomaly_indices_in_test) > 0:
                print(f"Found {len(anomaly_indices_in_test)} anomalies. Showing details for up to 5:")
                for i_sample, test_idx in enumerate(anomaly_indices_in_test[:5]):
                    print(f"\nAnomalous Sample (Index in Test Set: {test_idx}):")

                    true_binary_label_for_rl = y_test[test_idx] if test_idx < len(y_test) else 'N/A'

                    original_label_numeric = 'N/A'
                    original_label_str = 'N/A'
                    if y_test_original_labels is not None and test_idx < len(y_test_original_labels):
                        original_label_numeric = int(y_test_original_labels[test_idx])
                        original_label_str = str(original_label_numeric) 
                        if class_label_names is not None and original_label_numeric >= 0 and original_label_numeric < len(class_label_names):
                            original_label_str = class_label_names[original_label_numeric]
                    
                    print(f"  True Original Label: {original_label_str} (Numeric): {original_label_numeric})") # y_test is already flat
                    print(f"  True Binary Label (for RL task): {true_binary_label_for_rl}")
                    print(f"  Predicted Label: Anomaly (1)")
                    
                    if len(test_sample_scores) > test_idx :
                        print(f"  Anomaly Score (from model/feature): {test_sample_scores[test_idx]:.4f} (Threshold: {final_threshold:.4f})")

                    print(f"  Feature Values (from X_test, potentially after FE):")
                    if feature_names_rl and len(feature_names_rl) == X_test.shape[1] and X_test.ndim == 2:
                        for feature_idx in range(X_test.shape[1]):
                            print(f"    {feature_names_rl[feature_idx]}: {X_test[test_idx, feature_idx]:.4f}")
                    elif X_test.ndim == 2 and X_test.shape[1] > 0: # Fallback if feature_names are problematic but X_test is valid
                        for feature_idx in range(X_test.shape[1]):
                            print(f"    Feature {feature_idx}: {X_test[test_idx, feature_idx]:.4f}")
                    else:
                        print("    Feature values cannot be displayed (X_test shape or feature_names_rl issue).")
            else:
                print("No anomalies detected in the test set by the RL agent.")
        else:
            print("Cannot detail anomalous samples as y_pred is not valid for X_test.")

        print("\nRL Visualization complete! Check the 'plots' directory for detailed anomaly visualizations.")
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
