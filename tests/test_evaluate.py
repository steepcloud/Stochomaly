import unittest
import numpy as np
from trainer.evaluate import Evaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator()
        # synthetic data
        np.random.seed(42)
        self.X = np.random.randn(100, 10)
        self.y = np.random.randint(0, 2, 100)

    def test_optimal_threshold_finding(self):
        # test threshold optimization
        scores = np.random.random(100)
        threshold = self.evaluator._find_optimal_threshold(self.y, scores)
        self.assertTrue(0 < threshold < 1)

    def test_cross_validation(self):
        # mock trainer methods to avoid actual training
        self.evaluator.trainer.train = lambda X, y: None
        self.evaluator.trainer.predict = lambda X: np.random.random(len(X))

        # test cross-validation returns expected metrics
        cv_metrics = self.evaluator.cross_validate(self.X, self.y, n_splits=3)
        self.assertIn('accuracy', cv_metrics)
        self.assertIn('f1', cv_metrics)
        self.assertIn('roc_auc', cv_metrics)
        self.assertEqual(len(cv_metrics['f1']), 3)