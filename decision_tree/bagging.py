import typing as t

import numpy as np

from .decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    accuracy,
    most_frequent,
)


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        min_samples: int = 1,
        feature_names: t.Optional[t.List[str]] = None,
        target_names: t.Optional[t.List[str]] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feature_names = feature_names
        self.target_names = target_names
        self.estimators = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples = X.shape[0]
        # Bagging: sampling with replacement n_estimators
        # datasets of size n_samples
        self.estimators = []
        indices = np.random.randint(
            low=0, high=n_samples, size=(self.n_estimators, n_samples)
        )
        for i in range(self.n_estimators):
            idx = indices[i]
            X_est, y_est = X[idx], y[idx]

            estimator = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                feature_names=self.feature_names,
                target_names=self.target_names,
            )
            estimator.fit(X_est, y_est)
            self.estimators.append(estimator)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack(
            [estimator.predict(X) for estimator in self.estimators], axis=0
        )
        return np.array([most_frequent(preds[:, i]) for i in range(preds.shape[1])])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return accuracy(y, preds)
