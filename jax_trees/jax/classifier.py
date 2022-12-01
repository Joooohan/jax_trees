from __future__ import annotations

from functools import partial
from typing import Dict, List

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
    counts = jnp.bincount(y.astype(jnp.int8), weights=mask, length=n_classes)
    return jnp.nanargmax(counts)


def accuracy(y_hat: jnp.array, y: jnp.array) -> jnp.array:
    return jnp.mean(y_hat == y)


@register_pytree_node_class
class DecisionTreeClassifier(DecisionTree):
    def __init__(
        self,
        n_classes: int,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
        nodes: Dict[int, List[TreeNode]] = None,
    ):
        self.n_classes = n_classes

        super().__init__(
            min_samples=min_samples,
            max_depth=max_depth,
            max_splits=max_splits,
            loss_fn=partial(entropy, n_classes=n_classes),
            value_fn=partial(most_frequent, n_classes=n_classes),
            score_fn=accuracy,
            nodes=nodes,
        )

    def tree_flatten(self):
        children = [self.nodes]
        aux_data = {
            "min_samples": self.min_samples,
            "max_depth": self.max_depth,
            "max_splits": self.max_splits,
            "n_classes": self.n_classes,
        }
        return (children, aux_data)
