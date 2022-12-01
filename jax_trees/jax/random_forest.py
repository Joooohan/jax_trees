from __future__ import annotations

from functools import partial
from typing import Any, Optional, Type

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.tree_util import register_pytree_node_class

from .classifier import DecisionTreeClassifier
from .core import DecisionTree
from .regressor import DecisionTreeRegressor


class RandomForest:
    """Base class for random forests."""

    def __init__(
        self,
        n_estimators: int,
        base_class: Type[DecisionTree],
        predictors: DecisionTree,
        **kwargs: Any,
    ):
        self.base_class = base_class
        self.base_model = base_class(
            **kwargs,
        )
        self.n_estimators = n_estimators
        self.predictors = predictors

        self.aux_data = {
            "n_estimators": n_estimators,
        }
        self.aux_data.update(**kwargs)

    def tree_flatten(self):
        children = [self.predictors]
        return (children, self.aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (predictors,) = children
        return cls(**aux_data, predictors=predictors)

    @jit
    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> RandomForest:
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
            self.base_class.fit, in_axes=[None, None, None, 0]
        )(self.base_model, X, y, mask)
        return self

    @jit
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.predictors is None:
            raise ValueError("The model is not fitted.")

        preds = vmap(self.base_class.predict, in_axes=[0, None])(
            self.predictors, X
        )
        mask = jnp.ones_like(preds)
        return vmap(self.base_model.value_fn, in_axes=1)(preds, mask)

    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        preds = self.predict(X)
        return self.base_model.score_fn(preds, y)


@register_pytree_node_class
class RandomForestClassifier(RandomForest):
    def __init__(
        self,
        n_classes: int,
        n_estimators: int = 100,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
        predictors: Optional[DecisionTreeClassifier] = None,
    ):
        super().__init__(
            n_classes=n_classes,
            n_estimators=n_estimators,
            min_samples=min_samples,
            max_depth=max_depth,
            max_splits=max_splits,
            base_class=DecisionTreeClassifier,
            predictors=predictors,
        )


@register_pytree_node_class
class RandomForestRegressor(RandomForest):
    def __init__(
        self,
        n_estimators: int = 100,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
        predictors: Optional[DecisionTreeRegressor] = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            min_samples=min_samples,
            max_depth=max_depth,
            max_splits=max_splits,
            base_class=DecisionTreeRegressor,
            predictors=predictors,
        )
