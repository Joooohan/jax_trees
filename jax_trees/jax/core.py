from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List

import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.tree_util import register_pytree_node_class

from .utils import make_split_node_function, split_mask


@register_pytree_node_class
class TreeNode:
    """Class representing a node in the tree.

    For jitting to be polyvalent, the same node structure should be used to
    represent nodes, leaves and phantom nodes.

    The boolean `is_leaf` indicates if the node is a leaf and should be used to
    make a prediction.

    If the `mask` only contains zeros, then the node is actually a phantom node,
    that is, positionned below a leaf node and is not used.
    """

    def __init__(
        self,
        mask: jnp.ndarray,
        split_value: float = jnp.nan,
        split_col: int = -1,
        is_leaf: bool = True,
        leaf_value: float = jnp.nan,
        score: float = jnp.nan,
    ):
        self.mask = mask
        self.split_value = split_value
        self.split_col = split_col
        self.is_leaf = is_leaf
        self.leaf_value = leaf_value
        self.score = score

    def tree_flatten(self):
        children = (
            self.mask,
            self.split_value,
            self.split_col,
            self.is_leaf,
            self.leaf_value,
            self.score,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> TreeNode:
        return cls(*children)

    def show(self, rank: int) -> str:
        text = f"n={int(jnp.sum(self.mask[rank]))}\n"
        text += f"loss={self.score[rank]:.2f}\n"
        if self.is_leaf[rank]:
            text += f"value {self.leaf_value[rank]}"
        else:
            text += f"feature {self.split_col[rank]} >= {self.split_value[rank]:.2f}"
        return text


@register_pytree_node_class
class DecisionTree:
    def __init__(
        self,
        min_samples: int,
        max_depth: int,
        max_splits: int,
        loss_fn: Callable,
        value_fn: Callable,
        score_fn: Callable,
        nodes: Dict[int, List[TreeNode]] = None,
    ):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.max_splits = max_splits
        self.score_fn = score_fn
        self.loss_fn = loss_fn
        self.value_fn = value_fn
        self.nodes = nodes
        self.split_node = make_split_node_function(self.loss_fn)

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

    @jit
    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        mask: jnp.ndarray = None,
    ) -> TreeNode:
        """Fit the model to the data.

        Since this function is functionally pure, the fitted model is returned
        as a result.
        """
        n_samples = X.shape[0]
        if mask is None:
            mask = jnp.ones((n_samples,))

        masks = jnp.stack([mask], axis=0)
        self.nodes = defaultdict(list)

        def split_node(carry, x):
            depth, mask = x
            score = self.loss_fn(y, mask)
            value = self.value_fn(y, mask)
            (
                left_mask,
                right_mask,
                split_value,
                split_col,
            ) = self.split_node(X, y, mask, self.max_splits)

            is_leaf = jnp.maximum(depth + 1 - self.max_depth, 0) + jnp.maximum(
                self.min_samples + 1 - jnp.sum(mask), 0
            )
            is_leaf = jnp.minimum(is_leaf, 1).astype(jnp.int8)

            # zero-out child masks if current node is a leaf
            left_mask *= 1 - is_leaf
            right_mask *= 1 - is_leaf

            node = TreeNode(
                mask=mask,
                split_value=split_value,
                split_col=split_col,
                is_leaf=is_leaf,
                leaf_value=value,
                score=score,
            )
            children_mask = jnp.stack([left_mask, right_mask], axis=0)
            return carry, (children_mask, node)

        for depth in range(self.max_depth + 1):
            depths = depth * jnp.ones((masks.shape[0],))
            _, (next_masks, nodes) = lax.scan(
                f=split_node,
                init=None,
                xs=(depths, masks),
            )
            self.nodes[depth] = nodes
            masks = jnp.reshape(next_masks, (-1, n_samples))

        return self

    @jit
    def predict(
        self,
        X: jnp.ndarray,
        mask: jnp.ndarray = None,
    ) -> jnp.ndarray:
        X = X.astype("float32")
        n_samples = X.shape[0]

        if mask is None:
            mask = jnp.ones((n_samples,))

        if self.nodes is None:
            raise ValueError("The model is not fitted.")

        @vmap
        def split_and_predict(node, mask):
            left_mask, right_mask = split_mask(
                node.split_value,
                X.at[:, node.split_col].get(),
                mask,
            )
            predictions = jnp.where(
                mask * node.is_leaf, node.leaf_value, jnp.nan
            )
            child_mask = jnp.stack([left_mask, right_mask], axis=0)
            return child_mask, predictions

        predictions = []
        level_masks = jnp.stack([mask], axis=0)
        for depth in range(self.max_depth + 1):
            next_masks, level_predictions = split_and_predict(
                self.nodes[depth], level_masks
            )
            level_masks = jnp.reshape(next_masks, (-1, n_samples))
            predictions.append(jnp.nansum(level_predictions, axis=0))

        return jnp.nansum(jnp.stack(predictions, axis=0), axis=0)

    def score(self, X: jnp.DeviceArray, y: jnp.DeviceArray) -> float:
        preds = self.predict(X)
        return self.score_fn(preds, y)
