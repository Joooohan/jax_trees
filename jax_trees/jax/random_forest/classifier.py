from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap

from ..classifier import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self,
        n_classes: int,
        n_estimators: int = 100,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
    ):
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.max_splits = max_splits
        self.estimators = None

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        X = X.astype("float32")
        y = y.astype("int16")
        self.n_classes = jnp.size(jnp.bincount(y))

        self.estimators = []
        key = jax.random.PRNGKey(seed=0)
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            key, subkey = jax.random.split(key)
            model = DecisionTreeClassifier(
                n_classes=self.n_classes,
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                max_splits=self.max_splits,
            )
            idx = jax.random.randint(
                subkey, shape=(n_samples,), minval=0, maxval=n_samples
            )
            fitted_model = model.fit(X, y, jnp.bincount(idx, length=n_samples))
            self.estimators.append(fitted_model)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.estimators is None:
            raise ValueError("The model is not fitted.")

        X = X.astype("float32")
        preds = jnp.stack(
            [estimator.predict(X) for estimator in self.estimators],
            axis=0,
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
