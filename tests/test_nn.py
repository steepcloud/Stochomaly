import numpy as np
from trainer.train import Trainer
from data.preprocess import load_data, preprocess_data

def test_neural_network():
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y, scaler_type='minmax')

    # Split training data into training and validation sets for early stopping
    val_size = int(0.2 * len(X_train))  # 20% for validation
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train, y_train = X_train[val_size:], y_train[val_size:]

    schedulers = [
        {
            "name": "StepLR",
            "params": {"step_size": 10,
                       "gamma": 0.1
                       }
        },
        {
            "name": "ExponentialLR",
            "params": {"gamma": 0.9} # Decay rate per epoch
        },
        {
            "name": "ReduceLROnPlateau",
            "params": {"mode": "min", # Reduce LR when monitored value stops decreasing
                       "factor": 0.1, # Factor to reduce learning rate by
                       "patience": 10, # Number of epochs with no improvement after which LR will be reduced
                       "threshold": 1e-4, # Threshold for measuring improvement
                       "min_lr": 1e-6 # Minimum learning rate
                       }
        },
        {
            "name": "CosineAnnealingLR",
            "params": {"T_max": 50, # Maximum number of iterations/epochs
                       "eta_min": 0 # Minimum learning rate
                       }
        }
    ]

    results = {}

    for scheduler_config in schedulers:
        print(f"\nTesting {scheduler_config['name']} scheduler...")

        # Initialize trainer with current scheduler
        trainer = Trainer(
            input_size=X_train.shape[1],
            hidden_size=4,
            output_size=1,
            activation="relu",
            optimizer="adam",
            learning_rate=0.01,
            weight_decay=0.001,
            momentum=0.9,
            dropout_rate=0.3,
            early_stopping_patience=20,
            early_stopping_min_improvement=0.001,
            scheduler_type=scheduler_config['name'],
            scheduler_params=scheduler_config['params']
        )

        # Train model
        loss_history = trainer.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=1000,
            batch_size=1
        )

        # Evaluate model
        train_predictions = np.round(trainer.predict(X_train))
        test_predictions = np.round(trainer.predict(X_test))

        # Convert predictions to binary (classification case)
        train_predictions = np.round(train_predictions)
        test_predictions = np.round(test_predictions)

        train_accuracy = np.mean(train_predictions == y_train)
        test_accuracy = np.mean(test_predictions == y_test)

        # Store results
        results[scheduler_config['name']] = {
            'train_accuracy': train_accuracy * 100,
            'test_accuracy': test_accuracy * 100,
            'epochs_trained': len(loss_history),
            'stopped_early': trainer.stopped_early
        }

        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Epochs trained: {len(loss_history)}")
        if trainer.stopped_early:
            print("Training stopped early due to early stopping.")

    # Print summary of all results
    print("\n=== SUMMARY OF RESULTS ===")
    for scheduler, metrics in results.items():
        print(f"\n{scheduler}:")
        print(f"  Train Accuracy: {metrics['train_accuracy']:.2f}%")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.2f}%")
        print(f"  Epochs trained: {metrics['epochs_trained']}")
        print(f"  Stopped early: {metrics['stopped_early']}")

if __name__ == "__main__":
    test_neural_network()
