import numpy as np
from trainer.train import Trainer
from data.preprocess import load_data, preprocess_data

def test_neural_network():
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y, scaler_type='minmax')

    # Initialize trainer
    trainer = Trainer(
        input_size=2,
        hidden_size=4,
        output_size=1,
        activation="relu",
        optimizer="adam",
        learning_rate=0.01
    )

    # Train the model
    trainer.train(X_train, y_train, epochs=1000, batch_size=1)

    # Test predictions
    train_predictions = trainer.predict(X_train)
    train_accuracy = np.mean(np.round(train_predictions) == y_train)

    test_predictions = trainer.predict(X_test)
    test_accuracy = np.mean(np.round(test_predictions) == y_test)

    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_neural_network()