import jax.numpy as jnp
from jax import grad

from ..regressor import DecisionTreeRegressor, r2_score


def mse(y_hat: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Mean Square Error."""
    return jnp.mean(jnp.square(y_hat - y))


class GradientBoostedRegressor:
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
        self.loss = mse
        self.grad_loss = grad(self.loss)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        X = X.astype("float32")
        y = y.astype("float32")

        self.base_value = jnp.mean(y)

        current_preds = self.base_value * jnp.ones_like(y)
        self.estimators = []
        for _ in range(self.n_estimators):
            weak_learner = DecisionTreeRegressor(
                min_samples=self.min_samples,
                max_depth=self.max_depth,
                max_splits=self.max_splits,
            )
            residuals = -self.grad_loss(current_preds, y)
            weak_learner.fit(X, residuals)

            current_preds += self.learning_rate * weak_learner.predict(X)
            self.estimators.append(weak_learner)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.base_value is None:
            raise ValueError("The model is not fitted.")

        X = X.astype("float32")
        weak_preds = [weak_learner.predict(X) for weak_learner in self.estimators]
        preds_sum = jnp.sum(jnp.stack(weak_preds, axis=0), axis=0)
        return self.base_value + self.learning_rate * preds_sum

    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        preds = self.predict(X)
        return r2_score(y, preds)
