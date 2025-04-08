import numpy as np
import os
from trainer.train import Trainer
from data.preprocess import load_data, preprocess_data, DataLoader

def test_neural_network(data_source="xor", **data_params):
    # Load and preprocess data
    print(f"\nTesting with {data_source} dataset...")

    X, y = load_data(source=data_source, **data_params)

    # for datasets with more than 2 classes, we need binary targets for this test
    if data_source == "sklearn" and y.max() > 1:
        # convert to binary problem (0 vs rest)
        y = (y > 0).astype(int)

    X_train, X_test, y_train, y_test = preprocess_data(X, y, scaler_type='minmax')

    # Split training data into training and validation sets for early stopping
    val_size = max(1, int(0.2 * len(X_train)))  # 20% for validation
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

    # special case for XOR
    xor_batch_size = None
    if data_source == "xor":
        xor_batch_size = 4

    results = {}

    for scheduler_config in schedulers:
        print(f"\nTesting {scheduler_config['name']} scheduler...")

        # Initialize trainer with current scheduler
        trainer = Trainer(
            input_size=X_train.shape[1], # automatically adopts to data dimensions
            hidden_size=max(4, X_train.shape[1]), # scale hidden size with input features
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
            scheduler_params=scheduler_config['params'],
            use_batch_norm=False,
            use_bayesian=True,
            kl_weight=1.0
        )

        # Train model
        loss_history = trainer.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=100, # reduced for larger datasets
            batch_size=32 if xor_batch_size is None else 4, # using larger batches for bigger data
            n_samples=5
        )

        # Evaluate model
        train_predictions = np.round(trainer.predict(X_train))
        test_predictions = np.round(trainer.predict(X_test))

        '''
        # Convert predictions to binary (classification case)
        train_predictions = np.round(train_predictions)
        test_predictions = np.round(test_predictions)
        '''

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
        #print(f"Epochs trained: {len(loss_history)}")
        #if trainer.stopped_early:
        #    print("Training stopped early due to early stopping.")

    #print("\n=== SUMMARY OF RESULTS ===")
    #for scheduler, metrics in results.items():
    #    print(f"\n{scheduler}:")
    #    print(f"  Train Accuracy: {metrics['train_accuracy']:.2f}%")
    #    print(f"  Test Accuracy: {metrics['test_accuracy']:.2f}%")
    #    print(f"  Epochs trained: {metrics['epochs_trained']}")
    #    print(f"  Stopped early: {metrics['stopped_early']}")

    return results

if __name__ == "__main__":

    # original XOR data
    print("\n\n=== TESTING XOR DATASET ===")
    xor_results = test_neural_network(data_source="xor")

    # testing with sklearn iris dataset
    print("\n\n=== TESTING IRIS DATASET ===")
    iris_results = test_neural_network(
        data_source="sklearn",
        dataset_name="iris"
    )

    # testing with breast cancer dataset
    print("\n\n=== TESTING BREAST CANCER DATASET ===")
    cancer_results = test_neural_network(
        data_source="sklearn",
        dataset_name="breast_cancer"
    )

    '''
    # testing with CSV (if you have a CSV file)
    csv_path = "data/example.csv"
    if os.path.exists(csv_path):
        print(f"\n\n=== TESTING CSV DATASET: {csv_path} ===")
        csv_results = test_neural_network(
            data_source="csv",
            filepath=csv_path,
            target_col="target"
        )
    '''

    print("\n\n=== OVERALL COMPARISON ===")
    datasets = {
        "XOR": xor_results,
        "Iris": iris_results,
        "Breast Cancer": cancer_results
    }

    '''
    if 'csv_results' in locals():
        datasets["CSV"] = csv_results
    '''

    # printing the best scheduler for each dataset
    for dataset_name, results in datasets.items():
        best_scheduler = max(results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n{dataset_name} Dataset - Best Scheduler: {best_scheduler[0]}")
        print(f"  Test Accuracy: {best_scheduler[1]['test_accuracy']:.2f}%")
        print(f"  Epochs trained: {best_scheduler[1]['epochs_trained']}")

    #test_neural_network()
