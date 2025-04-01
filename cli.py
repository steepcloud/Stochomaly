import argparse
import numpy as np
from trainer.train import Trainer
from data.preprocess import load_data, preprocess_data


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

    #X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    #y = np.array([[0], [1], [1], [0]])
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(
        X, y,
        scaler_type=args.scaler
    )

    #trainer = Trainer(
    #    activation=args.activation,
    #    optimizer=args.optimizer,
    #    hidden_size=args.hidden_size,
    #    learning_rate=args.lr
    #)
    trainer = Trainer(
        input_size=X_train.shape[1],
        hidden_size=args.hidden_size,
        output_size=1,
        activation=args.activation,
        optimizer=args.optimizer,
        learning_rate=args.lr
    )

    if args.load_model:
        trainer.load_model(args.load_model)

    if args.train:
        #trainer.train(X, y, epochs=args.epochs, batch_size=args.batch_size)
        trainer.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

        if args.save_model:
            trainer.save_model(args.save_model)

        # evaluate on both train and test sets
        train_predictions = trainer.predict(X_train)
        train_accuracy = np.mean(np.round(train_predictions) == y_train)

        test_predictions = trainer.predict(X_test)
        test_accuracy = np.mean(np.round(test_predictions) == y_test)

        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


    else:
        # make predictions on both sets
        #predictions = trainer.predict(X)
        #print("Predictions:\n", predictions)
        train_predictions = trainer.predict(X_train)
        test_predictions = trainer.predict(X_test)

        print("Training Set Predictions:")
        print(np.round(train_predictions))
        print("\nTest Set Predictions:")
        print(np.round(test_predictions))

if __name__ == "__main__":
    main()
