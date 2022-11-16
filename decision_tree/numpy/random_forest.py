import typing as t
from multiprocessing import Pool

import numpy as np

from .trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    accuracy,
    most_frequent,
    r2_score,
)


def fit_decision_tree_classifier(data) -> DecisionTreeClassifier:
    X, y, max_depth, min_samples = data
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples=min_samples,
    )
    n_samples = X.shape[0]
    idx = np.random.randint(low=0, high=n_samples, size=(n_samples,))
    model.fit(X[idx], y[idx])
    return model


def fit_decision_tree_regressor(data) -> DecisionTreeRegressor:
    X, y, max_depth, min_samples = data
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples=min_samples,
    )
    n_samples = X.shape[0]
    idx = np.random.randint(low=0, high=n_samples, size=(n_samples,))
    model.fit(X[idx], y[idx])
    return model


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
        def params():
            for _ in range(self.n_estimators):
                yield [X, y, self.max_depth, self.min_samples]

        with Pool() as p:
            self.estimators = p.map(fit_decision_tree_classifier, params())

        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack(
            [estimator.predict(X) for estimator in self.estimators], axis=0
        )
        return np.array(
            [most_frequent(preds[:, i]) for i in range(preds.shape[1])]
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return accuracy(y, preds)


class RandomForestRegressor:
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
        def params():
            for _ in range(self.n_estimators):
                yield [X, y, self.max_depth, self.min_samples]

        with Pool() as p:
            self.estimators = p.map(fit_decision_tree_classifier, params())

        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack(
            [estimator.predict(X) for estimator in self.estimators], axis=0
        )
        return np.mean(preds, axis=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return r2_score(y, preds)
