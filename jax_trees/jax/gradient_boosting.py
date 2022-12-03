import jax
import jax.numpy as jnp
from jax import grad, lax, nn, vmap

from .regressor import DecisionTreeRegressor


def mean_square_error(y_hat: jnp.ndarray, y: jnp.ndarray) -> float:
    return jnp.mean(jnp.square(y_hat - y))


def categorical_cross_entropy(logits: jnp.ndarray, y: jnp.ndarray) -> float:
    """`y` is the one-hot encoded target."""
    return -jnp.mean(jnp.sum(nn.log_softmax(logits) * y, axis=1))


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


class GradientBoostedClassifier:
    def __init__(
        self,
        n_classes: int,
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
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_class = DecisionTreeRegressor
        self.base_model = self.base_class(**self.kwargs)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        y_oh = nn.one_hot(y, num_classes=self.n_classes)
        counts = jnp.sum(y_oh, axis=0)
        probs = counts / jnp.sum(counts)
        log_probs = jnp.log(probs)

        self.base_value = jnp.expand_dims(log_probs, axis=0)
        n_samples = X.shape[0]
        init_preds = self.base_value * jnp.ones((n_samples, 1))

        def fit_weak_learner(preds, _):
            model = self.base_class(**self.kwargs)
            residuals = -grad(categorical_cross_entropy)(preds, y_oh)
            fitted_model = vmap(
                DecisionTreeRegressor.fit, in_axes=[None, None, 1]
            )(model, X, residuals)
            weak_preds = vmap(
                DecisionTreeRegressor.predict, in_axes=[0, None], out_axes=1
            )(fitted_model, X)
            next_preds = preds + self.learning_rate * weak_preds
            return next_preds, fitted_model

        _, predictors = lax.scan(
            f=fit_weak_learner,
            init=init_preds,
            xs=jnp.arange(self.n_estimators),
        )

        self.predictors = predictors
        return self

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.base_value is None:
            raise ValueError("Model is not fitted.")

        weak_preds = vmap(
            vmap(DecisionTreeRegressor.predict, in_axes=[0, None]),
            in_axes=[0, None],
        )(self.predictors, X)
        logits = jnp.transpose(self.base_value) + self.learning_rate * jnp.sum(
            weak_preds, axis=0
        )
        key = jax.random.PRNGKey(seed=0)
        return jax.random.categorical(key, logits, axis=0)

    def score(self, X: jnp.DeviceArray, y: jnp.DeviceArray) -> float:
        preds = self.predict(X)
        return jnp.mean(preds == y)
