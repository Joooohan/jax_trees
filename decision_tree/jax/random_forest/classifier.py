from functools import partial
from multiprocessing import Pool

import jax
import jax.numpy as jnp
from jax import vmap

from ..classifier import DecisionTreeClassifier


def fit_decision_tree_classifier(data) -> DecisionTreeClassifier:
    X, y, max_depth, min_samples, max_splits, key = data
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples=min_samples,
        max_splits=max_splits,
    )
    n_samples = X.shape[0]
    idx = jax.random.randint(key, shape=(n_samples,), minval=0, maxval=n_samples)
    model.fit(X[idx], y[idx])
    return model


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
    ):
        self.n_estimators = n_estimators
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.max_splits = max_splits
        self.estimators = []

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        X = X.astype("float32")
        y = y.astype("int16")
        self.n_classes = jnp.size(jnp.bincount(y))

        def params():
            key = jax.random.PRNGKey(seed=0)
            for _ in range(self.n_estimators):
                key, subkey = jax.random.split(key)
                yield [X, y, self.max_depth, self.min_samples, self.max_splits, subkey]

        with Pool() as p:
            self.estimators = p.map(fit_decision_tree_classifier, params())

        return

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        X = X.astype("float32")
        preds = jnp.stack(
            [estimator.predict(X) for estimator in self.estimators], axis=0
        )
        batched_most_frequent = vmap(
            partial(jnp.bincount, length=self.n_classes), in_axes=(1,)
        )
        return jnp.argmax(
            batched_most_frequent(preds),
            axis=1,
        )

    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        preds = self.predict(X)
        return jnp.mean(preds == y)
