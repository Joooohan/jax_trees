from __future__ import annotations

import math
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional

import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class

from .utils import make_split_node_function, split_mask


@register_pytree_node_class
class TreeNode:
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

    def __str__(self) -> str:
        text = f"n={jnp.sum(self.mask)}\n"
        text += f"loss={self.score:.2f}\n"
        if self.is_leaf:
            text += f"value {self.leaf_value}"
        else:
            text += f"feature {self.split_col} >= {self.split_value:.2f}"
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

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> None:
        X = X.astype("float32")
        y = y.astype("int16")

        if mask is None:
            mask = jnp.ones_like(y)

        self.nodes = self.jitted_fit(X, y, mask)

    @partial(jit, static_argnames="n_classes")
    def jitted_fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> TreeNode:
        to_split = [mask]
        nodes = defaultdict(list)

        for idx in range((2 ** (self.max_depth + 1)) - 1):
            # getting current node mask
            mask = to_split.pop(0)
            depth = int(math.log2(idx + 1))

            score = self.loss_fn(y, mask)
            value = self.value_fn(y, mask)

            (
                left_mask,
                right_mask,
                split_value,
                split_col,
            ) = self.split_node(X, y, mask, self.max_splits)

            is_leaf = jnp.array(
                depth >= self.max_depth or jnp.sum(mask) <= self.min_samples,
                dtype=jnp.int8,
            )

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
            nodes[depth].append(node)
            to_split.extend((left_mask, right_mask))

        return nodes

    def predict(self, X: jnp.ndarray) -> jnp.DeviceArray:
        X = X.astype("float32")
        mask = jnp.ones((X.shape[0],))
        if self.nodes is None:
            raise ValueError("The model is not fitted.")
        return self.jitted_predict(X, mask)

    @jit
    def jitted_predict(
        self,
        X: jnp.array,
        mask: jnp.array,
    ) -> jnp.array:
        predictions = jnp.nan * jnp.zeros((X.shape[0],))
        masks = defaultdict(list)
        masks[0].append(mask)
        for depth in range(self.max_depth + 1):
            for rank, _ in enumerate(self.nodes[depth]):
                current_mask = masks[depth][rank]
                current_node = self.nodes[depth][rank]

                predictions = jnp.where(
                    current_mask, current_node.leaf_value, predictions
                )

                left_mask, right_mask = split_mask(
                    current_node.split_value,
                    X[:, current_node.split_col],
                    current_mask,
                )

                masks[depth + 1].extend((left_mask, right_mask))

        return predictions

    def score(self, X: jnp.DeviceArray, y: jnp.DeviceArray) -> float:
        preds = self.predict(X)
        return self.score_fn(preds, y)
