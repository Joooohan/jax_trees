from functools import partial

import jax.numpy as jnp
from jax import grad, nn

from ..regressor import DecisionTreeRegressor, r2_score


def binary_cross_entropy(logits: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Binary cross entropy loss."""
    log_p = nn.log_sigmoid(logits)
    log_one_minus_p = nn.log_sigmoid(-logits)
    return -jnp.sum((y * log_p + (1 - y) * log_one_minus_p))


class GradientBoostedClassifier:
    def __init__(
        self,
        n_estimators: int = 10,
        learning_rate: float = 1e-2,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.max_splits = max_splits

        self.estimators = None
        self.base_value = None

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        X = X.astype("float32")
        y = y.astype("int16")

        n_samples = y.shape[0]
        log_probs = jnp.log(jnp.bincount(y) / n_samples)
        n_classes = jnp.size(log_probs)
        self.base_value = log_probs
        y_oh = nn.one_hot(y, n_classes)

        grad_loss = grad(binary_cross_entropy)

        current_preds = jnp.repeat(
            jnp.expand_dims(log_probs, axis=0),
            repeats=n_samples,
            axis=0,
        )
        self.estimators = []

        for _ in range(self.n_estimators):
            stage_estimators, stage_preds = [], []
            for col in range(n_classes):
                # At each stage we need to fit `n_classes` estimators
                residuals = -grad_loss(current_preds[:, col], y_oh[:, col])
                weak_learner = DecisionTreeRegressor(
                    min_samples=self.min_samples,
                    max_depth=self.max_depth,
                    max_splits=self.max_splits,
                )
                weak_learner.fit(X, residuals)
                stage_estimators.append(weak_learner)
                stage_preds.append(weak_learner.predict(X))

            stage_preds = jnp.stack(stage_preds, axis=1)
            current_preds += self.learning_rate * stage_preds
            self.estimators.append(stage_estimators)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.base_value is None:
            raise ValueError("The model is not fitted.")

        X = X.astype("float32")
        weak_preds = [
            jnp.stack(
                [weak_learner.predict(X) for weak_learner in stage_estimators], axis=1
            )
            for stage_estimators in self.estimators
        ]
        preds_sum = jnp.sum(jnp.stack(weak_preds, axis=0), axis=0)
        logits = self.base_value + self.learning_rate * preds_sum
        return jnp.nanargmax(logits, axis=1)

    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        preds = self.predict(X)
        return r2_score(y, preds)
