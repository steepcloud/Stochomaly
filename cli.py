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
    args = parser.parse_args()

    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y, scaler_type=args.scaler)

    # Initialize trainer with correct input size
    trainer = Trainer(
        input_size=X_train.shape[1],
        hidden_size=args.hidden_size,
        output_size=1,
        activation=args.activation,
        optimizer=args.optimizer,
        learning_rate=args.lr
    )

    # Load model if specified
    if args.load_model:
        if os.path.exists(args.load_model):
            trainer.load_model(args.load_model)
        else:
            print(f"Warning: Model file '{args.load_model}' not found. Proceeding without loading.")

    if args.train:
        # Train the model
        loss_history = trainer.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

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
