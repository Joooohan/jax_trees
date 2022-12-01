from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap

from .classifier import DecisionTreeClassifier, most_frequent


class RandomForestClassifier:
    def __init__(
        self,
        n_classes: int,
        n_estimators: int = 100,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
    ):
        self.base_model = DecisionTreeClassifier(
            n_classes=n_classes,
            min_samples=min_samples,
            max_depth=max_depth,
            max_splits=max_splits,
        )
        self.n_estimators = n_estimators
        self.predictors = None

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        n_samples = X.shape[0]
        key = jax.random.PRNGKey(seed=0)
        idx = jax.random.randint(
            key,
            shape=(self.n_estimators, n_samples),
            minval=0,
            maxval=n_samples,
        )
        # Bootstrap dataset by creating a mask, weighting each sample with a
        # positive integer, the mask's sum being equal to n_samples
        mask = vmap(partial(jnp.bincount, length=n_samples))(idx)
        self.predictors = vmap(
            DecisionTreeClassifier.fit, in_axes=[None, None, None, 0]
        )(self.base_model, X, y, mask)
        return self

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        preds = vmap(DecisionTreeClassifier.predict, in_axes=[0, None])(
            self.predictors, X
        )
        mask = jnp.ones_like(preds)
        return vmap(self.base_model.value_fn, in_axes=1)(
            preds.astype(jnp.int8), mask
        )

    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        preds = self.predict(X)
        return self.base_model.score_fn(preds, y)
