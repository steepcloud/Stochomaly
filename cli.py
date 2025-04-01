import argparse
import numpy as np
from nn_core.neural_network import NeuralNetwork

def main():
    parser = argparse.ArgumentParser(description="Train and test a neural network.")
    parser.add_argument("--train", action="store_true", help="Train the neural network")
    args = parser.parse_args()

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)

    if args.train:
        nn.train(X, y, epochs=1000)
    else:
        predictions = nn.predict(X)
        print("Predictions:", predictions)

if __name__ == "__main__":
    main()
