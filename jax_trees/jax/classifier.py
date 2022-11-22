from __future__ import annotations

import math
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

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
    mask: jnp.ndarray,
    max_splits: int,
    n_classes: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, int]:
    """The algorithm does the following:

    1. Generate split points candidates (N_SPLITS, N_COLS) matrix
    2. For each split point, compute the split score -> (N_SPLITS, N_COLS)
    3. Select the point with the lowest score
    4. Generate two new masks for left and right children nodes
    """
    print("Tracing `split_node`")
    points = split_points(X, mask, max_splits)
    scores = compute_all_scores(X, y, mask, points, n_classes)

    split_row, split_col = jnp.unravel_index(
        jnp.nanargmin(scores), scores.shape
    )
    split_value = points[split_row, split_col]
    left_mask, right_mask = split_mask(split_value, X[:, split_col], mask)

    return left_mask, right_mask, split_value, split_col


@partial(jit, static_argnames=["n_classes"])
def most_frequent(y: jnp.ndarray, mask: jnp.ndarray, n_classes: int) -> int:
    counts = jnp.bincount(y, weights=mask, length=n_classes)
    return jnp.nanargmax(counts)


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
        self.nodes = None

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
        self.nodes = _fit(
            X,
            y,
            mask,
            n_classes,
            self.max_depth,
            self.min_samples,
            self.max_splits,
        )

    def predict(self, X) -> jnp.DeviceArray:
        X = X.astype("float32")
        mask = jnp.ones((X.shape[0],))
        if self.nodes is None:
            raise ValueError("The model is not fitted.")
        return _predict(X, mask, self.nodes, self.max_depth)

    def score(self, X: jnp.DeviceArray, y: jnp.DeviceArray) -> float:
        preds = self.predict(X)
        return jnp.mean(preds == y)


@partial(
    jit,
    static_argnames=["n_classes", "max_depth", "min_samples", "max_splits"],
)
def _fit(
    X: jnp.ndarray,
    y: jnp.ndarray,
    mask: jnp.ndarray,
    n_classes: int,
    max_depth: int,
    min_samples: int,
    max_splits: int,
) -> TreeNode:

    to_split = [mask]
    nodes = defaultdict(list)

    for idx in range((2 ** (max_depth + 1)) - 1):
        # getting current node mask
        mask = to_split.pop(0)
        depth = int(math.log2(idx + 1))

        score = entropy(y, mask, n_classes)
        value = most_frequent(y, mask, n_classes)

        (
            left_mask,
            right_mask,
            split_value,
            split_col,
        ) = split_node(X, y, mask, max_splits, n_classes)

        is_leaf = jnp.array(
            depth >= max_depth or jnp.sum(mask) <= min_samples,
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


def split_mask(
    value: float, col: jnp.ndarray, mask: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    left_mask = jnp.where(col >= value, mask, 0)
    right_mask = jnp.where(col < value, mask, 0)
    return left_mask, right_mask


# @jit
def _predict(
    X: jnp.array,
    mask: jnp.array,
    nodes: Dict[int, List[TreeNode]],
    max_depth: int,
) -> jnp.array:
    predictions = jnp.nan * jnp.zeros((X.shape[0],))
    to_visit = [(nodes[0][0], mask)]
    for depth in range(max_depth + 1):
        for rank, _ in enumerate(nodes[depth]):
            current_node, current_mask = to_visit.pop(0)
            if current_node is None:
                to_visit.append((None, None))
                to_visit.append((None, None))
            elif not current_node.is_leaf:
                left_mask, right_mask = split_mask(
                    current_node.split_value,
                    X[:, current_node.split_col],
                    current_mask,
                )
                left_node = nodes[depth + 1][2 * rank]
                right_node = nodes[depth + 1][2 * rank + 1]
                to_visit.append((left_node, left_mask))
                to_visit.append((right_node, right_mask))
            else:
                predictions = jnp.where(
                    current_mask, current_node.leaf_value, predictions
                )
                to_visit.append((None, None))
                to_visit.append((None, None))

    return predictions
