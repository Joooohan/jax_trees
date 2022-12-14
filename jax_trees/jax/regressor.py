from typing import Dict, List

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .core import DecisionTree, TreeNode


def variance(y: jnp.ndarray, mask: jnp.ndarray) -> float:
    """Weighted version of variance."""
    avg = jnp.average(y, weights=mask)
    return jnp.average(jnp.square(y - avg), weights=mask)


def average(y: jnp.ndarray, mask: jnp.ndarray) -> float:
    """Masked version of the mean."""
    return jnp.average(y, weights=mask)


def r2_score(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> float:
    """Score for regressors."""
    u = jnp.square(y_true - y_pred).sum()
    v = jnp.sum(jnp.square(y_true - jnp.mean(y_true)))
    return 1 - u / v


@register_pytree_node_class
class DecisionTreeRegressor(DecisionTree):
    def __init__(
        self,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
        nodes: Dict[int, List[TreeNode]] = None,
    ):
        super().__init__(
            min_samples=min_samples,
            max_depth=max_depth,
            max_splits=max_splits,
            loss_fn=variance,
            value_fn=average,
            score_fn=r2_score,
            nodes=nodes,
        )

    def tree_flatten(self):
        children = [self.nodes]
        aux_data = {
            "min_samples": self.min_samples,
            "max_depth": self.max_depth,
            "max_splits": self.max_splits,
        }
        return (children, aux_data)
