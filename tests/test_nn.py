import numpy as np
from trainer.train import Trainer
from data.preprocess import load_data, preprocess_data
import matplotlib.pyplot as plt

def test_neural_network():
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y, scaler_type='minmax')

    # Initialize trainer
    trainer = Trainer(
        input_size=X_train.shape[1],  # Ensure correct input size
        hidden_size=4,
        output_size=1,
        activation="relu",
        optimizer="adam",
        learning_rate=0.01,
        weight_decay=0.001,
        momentum=0.9,
        dropout_rate=0.3
    )

    # Train the model
    loss_history = trainer.train(X_train, y_train, epochs=1000, batch_size=1)

    # Test predictions
    train_predictions = trainer.predict(X_train)
    test_predictions = trainer.predict(X_test)

    # Convert predictions to binary (classification case)
    train_predictions = np.round(train_predictions)
    test_predictions = np.round(test_predictions)

    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)

    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_neural_network()
