from __future__ import annotations

from functools import partial
from typing import Callable, Dict, List, Optional

import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class

from .core import DecisionTree, TreeNode


@partial(jit, static_argnames=["n_classes"])
def entropy(y: jnp.ndarray, mask: jnp.ndarray, n_classes: int) -> float:
    """Shannon entropy in bits.

    Returns NaN if no samples.
    """
    n_samples = jnp.sum(mask)
    counts = jnp.bincount(y, weights=mask, length=n_classes)
    probs = counts / n_samples
    log_probs = probs * jnp.log2(probs)
    return -jnp.sum(jnp.where(probs <= 0.0, 0.0, log_probs))


@partial(jit, static_argnames=["n_classes"])
def most_frequent(y: jnp.ndarray, mask: jnp.ndarray, n_classes: int) -> int:
    counts = jnp.bincount(y, weights=mask, length=n_classes)
    return jnp.nanargmax(counts)


def accuracy(y_hat: jnp.array, y: jnp.array) -> jnp.array:
    return jnp.mean(y_hat == y)


@register_pytree_node_class
class DecisionTreeClassifier(DecisionTree):
    def __init__(
        self,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
        loss_fn: Callable = entropy,
        value_fn: Callable = most_frequent,
        score_fn: Callable = accuracy,
        nodes: Dict[int, List[TreeNode]] = None,
    ):
        super().__init__(
            min_samples=min_samples,
            max_depth=max_depth,
            max_splits=max_splits,
            loss_fn=loss_fn,
            value_fn=value_fn,
            score_fn=score_fn,
            nodes=nodes,
        )

    def tree_flatten(self):
        children = [self.nodes]
        aux_data = {
            "min_samples": self.min_samples,
            "max_depth": self.max_depth,
            "max_splits": self.max_splits,
            "score_fn": self.score_fn,
            "value_fn": self.value_fn,
            "loss_fn": self.loss_fn,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (nodes,) = children
        return cls(**aux_data, nodes=nodes)

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> None:
        X = X.astype("float32")
        y = y.astype("int16")
        n_classes = jnp.size(jnp.bincount(y))

        self.loss_fn = partial(self.loss_fn, n_classes=n_classes)
        self.value_fn = partial(self.value_fn, n_classes=n_classes)

        if mask is None:
            mask = jnp.ones_like(y)

        self.nodes = self.jitted_fit(X, y, mask)
