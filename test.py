from trainer.evaluate import Evaluator
from trainer.train import Trainer
import numpy as np

# 1. Prepare XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# 2. Split data (using more data for training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, shuffle=True
)

# 3. Train model
trainer = Trainer(input_size=2, hidden_size=4, output_size=1)
trainer.train(
    X_train,
    y_train,
    epochs=2000,
    batch_size=1
)
trainer.save_model('model.pkl')

# 4. Evaluate
evaluator = Evaluator('model.pkl')
metrics = evaluator.evaluate(X_test, y_test, threshold=0.5)