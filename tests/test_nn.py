# tests/test_nn.py
import numpy as np
from trainer.train import Trainer

def test_neural_network():
    # Sample dataset (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialize the trainer
    trainer = Trainer(
        input_size=2,
        hidden_size=4,
        output_size=1,
        activation="relu",
        optimizer="adam",
        learning_rate=0.01
    )

    # Train the neural network
    trainer.train(X, y, epochs=1000, batch_size=1)

    # Predict on the training data
    predictions = trainer.predict(X)
    predictions = np.round(predictions)  # Round predictions to 0 or 1

    # Calculate accuracy
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_neural_network()