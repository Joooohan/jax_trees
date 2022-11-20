from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import jit, vmap
from jax.tree_util import register_pytree_node

from .utils import split_mask, split_points


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


def compute_score(
    X_col: jnp.ndarray,
    y: jnp.ndarray,
    mask: jnp.ndarray,
    split_value: float,
    n_classes: int,
) -> float:
    """Compute the scores of data splits."""
    left_mask, right_mask = split_mask(split_value, X_col, mask)

    left_score = entropy(y, left_mask, n_classes)
    right_score = entropy(y, right_mask, n_classes)

    n_left = jnp.sum(left_mask)
    n_right = jnp.sum(right_mask)

    avg_score = (n_left * left_score + n_right * right_score) / (
        n_left + n_right
    )

    return avg_score


compute_column_scores = vmap(
    compute_score, in_axes=(None, None, None, 0, None)
)

compute_all_scores = vmap(
    compute_column_scores,
    in_axes=(1, None, None, 1, None),
    out_axes=1,
)


@partial(jit, static_argnames=["max_splits", "n_classes"])
def split_node(
    X: jnp.ndarray,
    y: jnp.ndarray,
    current_node: TreeNode,
    max_splits: int,
    n_classes: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, int]:
    """The algorithm does the following:

    1. Generate split points candidates (N_SPLITS, N_COLS) matrix
    2. For each split point, compute the split score -> (N_SPLITS, N_COLS)
    3. Select the point with the lowest score
    4. Generate two new masks for left and right children nodes
    """
    points = split_points(X, current_node.mask, max_splits)
    scores = compute_all_scores(X, y, current_node.mask, points, n_classes)

    split_row, split_col = jnp.unravel_index(
        jnp.nanargmin(scores), scores.shape
    )
    split_value = points[split_row, split_col]
    left_mask, right_mask = split_mask(
        split_value, X[:, split_col], current_node.mask
    )

    return TreeNode(left_mask), TreeNode(right_mask), split_value, split_col


@partial(jit, static_argnames=["n_classes"])
def most_frequent(y: jnp.ndarray, mask: jnp.ndarray, n_classes: int) -> int:
    counts = jnp.bincount(y, weights=mask, length=n_classes)
    return jnp.nanargmax(counts)


class TreeNode:
    def __init__(
        self,
        mask,
        split_value=None,
        split_col=None,
        left_node=None,
        right_node=None,
        is_leaf=True,
        leaf_value=None,
        score=None,
    ):
        self.mask = mask
        self.split_value = split_value
        self.split_col = split_col
        self.left_node = left_node
        self.right_node = right_node
        self.is_leaf = is_leaf
        self.leaf_value = leaf_value
        self.score = score

    def __str__(self) -> str:
        text = f"n={jnp.sum(self.mask)}\n"
        text += f"entropy={self.score:.2f}\n"
        if self.is_leaf:
            text += f"value {self.leaf_value}"
        else:
            text += f"feature {self.split_col} >= {self.split_value:.2f}"
        return text


def special_flatten(node: TreeNode):
    children = (
        node.mask,
        node.split_value,
        node.split_col,
        node.left_node,
        node.right_node,
        node.is_leaf,
        node.leaf_value,
        node.score,
    )
    aux_data = None
    return children, aux_data


def special_unflatten(aux_data, children) -> TreeNode:
    return TreeNode(*children)


register_pytree_node(
    TreeNode,
    special_flatten,
    special_unflatten,
)


class DecisionTreeClassifier:
    def __init__(
        self, min_samples: int = 2, max_depth: int = 4, max_splits: int = 25
    ):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.max_splits = max_splits
        self.root = None

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
        n_classes = jnp.size(jnp.bincount(y))
        self.root = TreeNode(mask)

        to_split = [self.root]
        for idx in range((2 ** (self.max_depth + 1)) - 1):
            current_node = to_split.pop(0)
            depth = int(math.log2(idx + 1))

            if current_node is None:
                to_split.extend((None, None))

            elif (
                depth < self.max_depth
                and jnp.sum(current_node.mask) > self.min_samples
            ):
                (
                    left_node,
                    right_node,
                    split_value,
                    split_col,
                ) = split_node(X, y, current_node, self.max_splits, n_classes)

                current_node.score = entropy(y, current_node.mask, n_classes)
                current_node.is_leaf = False
                current_node.split_value = split_value
                current_node.split_col = split_col
                current_node.left_node = left_node
                current_node.right_node = right_node
                to_split.extend((left_node, right_node))
            else:
                current_node.score = entropy(y, current_node.mask, n_classes)
                current_node.leaf_value = most_frequent(
                    y, current_node.mask, n_classes
                )
                to_split.extend((None, None))

    def predict(self, X) -> jnp.DeviceArray:
        X = X.astype("float32")
        mask = jnp.ones((X.shape[0],))
        if self.root is None:
            raise ValueError("The model is not fitted.")

        predictions = jnp.nan * jnp.zeros((X.shape[0],))
        to_visit = [(self.root, mask)]
        while len(to_visit) > 0:
            current_node, current_mask = to_visit.pop(0)
            if not current_node.is_leaf:
                left_mask, right_mask = split_mask(
                    current_node.split_value,
                    X[:, current_node.split_col],
                    current_mask,
                )
                to_visit.append((current_node.left_node, left_mask))
                to_visit.append((current_node.right_node, right_mask))
            else:
                predictions = jnp.where(
                    current_mask, current_node.leaf_value, predictions
                )

        return predictions

    def score(self, X: jnp.DeviceArray, y: jnp.DeviceArray) -> float:
        preds = self.predict(X)
        return jnp.mean(preds == y)
