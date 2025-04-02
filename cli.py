import argparse
import numpy as np
from trainer.train import Trainer
from data.preprocess import load_data, preprocess_data
import os


def main():
    parser = argparse.ArgumentParser(description="Train and test a neural network.")
    parser.add_argument("--train", action="store_true", help="Train the neural network")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function")
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
    parser.add_argument("--early-stopping-patience", type=int, default=20,
                        help="Number of epochs with no improvement after which training stops")
    parser.add_argument("--early-stopping-min-improvement", type=float, default=0.001,
                        help="Minimum change to qualify as an improvement for early stopping")

    # Scheduler-specific parameters
    # StepLR parameters
    parser.add_argument("--step-size", type=int, default=10,
                        help="Period of learning rate decay (for StepLR)")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Multiplicative factor of learning rate decay")

    # ReduceLROnPlateau parameters
    parser.add_argument("--mode", type=str, default="min", choices=["min", "max"],
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


    args = parser.parse_args()

    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y, scaler_type=args.scaler)

    # Split training data into training and validation sets
    val_size = int(0.2 * len(X_train)) # 20% for validation
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
            "mode": args.mode,
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
        optimizer=args.optimizer,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_improvement=args.early_stopping_min_improvement,
        scheduler_type=scheduler_type,
        scheduler_params=scheduler_params
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
                                     epochs=args.epochs, batch_size=args.batch_size)

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


if __name__ == "__main__":
    main()
