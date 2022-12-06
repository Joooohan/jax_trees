from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import grad, jit, lax, nn, vmap
from jax.tree_util import register_pytree_node_class

from .classifier import accuracy
from .regressor import DecisionTreeRegressor, r2_score


def mean_square_error(y_hat: jnp.ndarray, y: jnp.ndarray) -> float:
    return jnp.mean(jnp.square(y_hat - y))


def categorical_cross_entropy(logits: jnp.ndarray, y: jnp.ndarray) -> float:
    """`y` is the one-hot encoded target."""
    return -jnp.mean(jnp.sum(nn.log_softmax(logits) * y, axis=1))


class GradientBoostedMachine:
    def __init__(
        self,
        n_estimators: int,
        learning_rate: float,
        min_samples: int,
        max_depth: int,
        max_splits: int,
        loss: Callable,
        fit_wl: Callable,
        predict_wl: Callable,
        score_fn: Callable,
        base_value: Optional[jnp.ndarray] = None,
        predictors: Optional[jnp.ndarray] = None,
    ):
        self.base_value = base_value
        self.predictors = predictors

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tree_args = {
            "min_samples": min_samples,
            "max_depth": max_depth,
            "max_splits": max_splits,
        }
        self.loss = loss
        self.fit_wl = fit_wl
        self.predict_wl = predict_wl
        self.score_fn = score_fn

        # Used to serialize the model
        self.aux_data = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "min_samples": min_samples,
            "max_depth": max_depth,
            "max_splits": max_splits,
        }

    def tree_flatten(self):
        children = [self.predictors, self.base_value]
        return (children, self.aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (predictors, base_value) = children
        return cls(**aux_data, predictors=predictors, base_value=base_value)

    def preprocess(
        self, X: jnp.ndarray, y: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError

    @jit
    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        X, y, self.base_value = self.preprocess(X, y)

        n_samples = X.shape[0]
        init_preds = jnp.repeat(
            jnp.expand_dims(self.base_value, axis=0), repeats=n_samples, axis=0
        )

        def fit_weak_learner(preds, x):
            model = DecisionTreeRegressor(**self.tree_args)
            residuals = -grad(self.loss)(preds, y)
            fitted_model = self.fit_wl(model, X, residuals)
            weak_preds = self.predict_wl(fitted_model, X)
            next_preds = preds + self.learning_rate * weak_preds
            return next_preds, fitted_model

        _, predictors = lax.scan(
            f=fit_weak_learner,
            init=init_preds,
            xs=jnp.arange(self.n_estimators),
        )

        self.predictors = predictors
        return self

    @jit
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.base_value is None:
            raise ValueError("Model is not fitted.")

        weak_preds = vmap(self.predict_wl, in_axes=[0, None])(
            self.predictors, X
        )
        preds = jnp.expand_dims(
            self.base_value, axis=0
        ) + self.learning_rate * jnp.sum(weak_preds, axis=0)
        return preds

    def score(self, X: jnp.DeviceArray, y: jnp.DeviceArray) -> float:
        preds = self.predict(X)
        return self.score_fn(preds, y)


@register_pytree_node_class
class GradientBoostedRegressor(GradientBoostedMachine):
    def __init__(
        self,
        n_estimators: int = 10,
        learning_rate: float = 1e-2,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
        base_value: Optional[jnp.ndarray] = None,
        predictors: Optional[jnp.ndarray] = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples=min_samples,
            max_depth=max_depth,
            max_splits=max_splits,
            score_fn=r2_score,
            loss=mean_square_error,
            fit_wl=DecisionTreeRegressor.fit,
            predict_wl=DecisionTreeRegressor.predict,
            base_value=base_value,
            predictors=predictors,
        )

    def preprocess(
        self, X: jnp.ndarray, y: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        base_value = jnp.mean(y)
        return X, y, base_value


@register_pytree_node_class
class GradientBoostedClassifier(GradientBoostedMachine):
    def __init__(
        self,
        n_classes: int,
        n_estimators: int = 10,
        learning_rate: float = 1e-2,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
        base_value: Optional[jnp.ndarray] = None,
        predictors: Optional[jnp.ndarray] = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples=min_samples,
            max_depth=max_depth,
            max_splits=max_splits,
            score_fn=accuracy,
            loss=categorical_cross_entropy,
            fit_wl=vmap(DecisionTreeRegressor.fit, in_axes=[None, None, 1]),
            predict_wl=vmap(
                DecisionTreeRegressor.predict, in_axes=[0, None], out_axes=1
            ),
            base_value=base_value,
            predictors=predictors,
        )
        self.n_classes = n_classes
        self.aux_data["n_classes"] = n_classes

    def preprocess(
        self, X: jnp.ndarray, y: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        y_oh = nn.one_hot(y, num_classes=self.n_classes)
        counts = jnp.sum(y_oh, axis=0)
        probs = counts / jnp.sum(counts)
        log_probs = jnp.log(probs)
        return X, y_oh, log_probs

    @jit
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        logits = super().predict(X)
        key = jax.random.PRNGKey(seed=0)
        return jax.random.categorical(key, logits, axis=1)
