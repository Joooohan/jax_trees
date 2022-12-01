import jax.numpy as jnp
from jax import grad, lax, vmap

from .regressor import DecisionTreeRegressor


def mean_square_error(y_hat: jnp.ndarray, y: jnp.ndarray) -> float:
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
        self.kwargs = {
            "min_samples": min_samples,
            "max_depth": max_depth,
            "max_splits": max_splits,
        }
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_class = DecisionTreeRegressor
        self.base_model = self.base_class(**self.kwargs)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        self.base_value = jnp.mean(y)

        def fit_weak_learner(preds, x):
            model = self.base_class(**self.kwargs)
            residuals = -grad(mean_square_error)(preds, y)
            fitted_model = model.fit(X, residuals)
            next_preds = preds + self.learning_rate * fitted_model.predict(X)
            return next_preds, fitted_model

        _, predictors = lax.scan(
            f=fit_weak_learner,
            init=self.base_value * jnp.ones_like(y),
            xs=jnp.arange(self.n_estimators),
        )

        self.predictors = predictors
        return self

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.base_value is None:
            raise ValueError("Model is not fitted.")

        mask = jnp.ones((X.shape[0],))
        preds = vmap(self.base_class.predict, in_axes=[0, None, None])(
            self.predictors, X, mask
        )
        return self.base_value + self.learning_rate * jnp.sum(preds, axis=0)

    def score(self, X: jnp.DeviceArray, y: jnp.DeviceArray) -> float:
        preds = self.predict(X)
        return self.base_model.score_fn(preds, y)
