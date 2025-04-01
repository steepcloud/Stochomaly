import argparse
import numpy as np
from trainer.train import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train and test a neural network.")
    parser.add_argument("--train", action="store_true", help="Train the neural network")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer function")
    args = parser.parse_args()

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    trainer = Trainer(activation=args.activation, optimizer=args.optimizer)

    if args.train:
        trainer.train(X, y, epochs=args.epochs)
    else:
        predictions = trainer.predict(X)
        print("Predictions:\n", predictions)

if __name__ == "__main__":
    main()
