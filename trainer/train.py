import numpy as np
from nn_core.neural_network import NeuralNetwork

# Example dataset (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize and train the network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
nn.train(X, y, epochs=1000)

# Predictions
predictions = nn.predict(X)
print("Predictions:\n", predictions)
