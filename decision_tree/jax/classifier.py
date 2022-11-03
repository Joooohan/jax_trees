from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import jit, vmap

from .utils import split_points


@partial(jit, static_argnames=["n_classes"])
def entropy(y, mask, n_classes):
    """Shannon entropy in bits.

    NB: In jnp.bincounts, values greater than length-1 are discarded so we set
    values of `y` ignored by the mask to `n_classes` to discard them.
    """
    n_samples = jnp.sum(mask)
    counts = jnp.bincount(jnp.where(mask, y, n_classes), length=n_classes)
    probs = counts / n_samples
    log_probs = probs * jnp.log2(probs)
    return -jnp.sum(jnp.where(probs <= 0.0, 0.0, log_probs))


def compute_score(X_col, y, mask, split_value, n_classes):
    """Compute the scores of data splits."""
    left_mask = jnp.where(X_col >= split_value, mask, False)
    right_mask = jnp.where(X_col < split_value, mask, False)

    left_score = entropy(y, left_mask, n_classes)
    right_score = entropy(y, right_mask, n_classes)

    n_left = jnp.sum(left_mask)
    n_right = jnp.sum(right_mask)

    avg_score = (n_left * left_score + n_right * right_score) / (n_left + n_right)

    return avg_score


compute_column_scores = vmap(compute_score, in_axes=(None, None, None, 0, None))

compute_all_scores = vmap(
    compute_column_scores,
    in_axes=(1, None, None, 1, None),
    out_axes=1,
)


@partial(jit, static_argnames=["max_splits", "n_classes"])
def split_node(X, y, mask, max_splits: int, n_classes: int):
    """The algorithm does the following:

    1. Generate split points candidates (N_SPLITS, N_COLS) matrix
    2. For each split point, compute the split score -> (N_SPLITS, N_COLS)
    3. Select the point with the lowest score
    4. Generate two new masks for left and right children nodes
    """
    points = split_points(X, mask, max_splits)
    scores = compute_all_scores(X, y, mask, points, n_classes)

    split_row, split_col = jnp.unravel_index(jnp.nanargmin(scores), scores.shape)
    split_value = points[split_row, split_col]

    left_mask = jnp.where(X[:, split_col] >= split_value, mask, False)
    right_mask = jnp.where(X[:, split_col] < split_value, mask, False)
    return left_mask, right_mask, split_value, split_col


@partial(jit, static_argnames=["n_classes"])
def most_frequent(y, mask, n_classes):
    counts = jnp.bincount(jnp.where(mask, y, n_classes), length=n_classes)
    return jnp.nanargmax(counts)


class TreeNode:
    def __init__(
        self, X, y, mask, min_samples: int, depth: int, max_splits: int, n_classes: int
    ):
        self.n_samples = jnp.sum(mask)
        self.score = entropy(y, mask, n_classes)
        self.feature_names = None
        self.target_names = None

        if jnp.sum(mask) > min_samples and depth > 0:
            left_mask, right_mask, split_value, split_col = split_node(
                X, y, mask, max_splits, n_classes
            )
            self.is_leaf = False
            self.left_node = TreeNode(
                X, y, left_mask, min_samples, depth - 1, max_splits, n_classes
            )
            self.right_node = TreeNode(
                X, y, right_mask, min_samples, depth - 1, max_splits, n_classes
            )
            self.split_value = split_value
            self.split_col = split_col
        else:
            self.is_leaf = True
            self.value = most_frequent(y, mask, n_classes)

    def predict(self, X: jnp.DeviceArray, mask: jnp.DeviceArray) -> jnp.DeviceArray:
        if self.is_leaf:
            return jnp.where(mask, self.value, np.nan)
        else:
            left_mask = jnp.where(X[:, self.split_col] >= self.split_value, mask, False)
            right_mask = jnp.where(X[:, self.split_col] < self.split_value, mask, False)
            right_pred = self.right_node.predict(X, right_mask)
            left_pred = self.left_node.predict(X, left_mask)
            return jnp.where(
                left_mask, left_pred, jnp.where(right_mask, right_pred, np.nan)
            )

    def __repr__(self) -> str:
        text = f"n={self.n_samples}\n"
        text += f"entropy {self.score:.2f}\n"
        if not self.is_leaf:
            if self.feature_names is not None:
                col_name = self.feature_names[self.split_col]
            else:
                col_name = f"feature {self.split_col}"
            text += f"{col_name} >= {self.split_value:.2f}"
        else:
            if self.target_names is not None:
                target_name = self.target_names[self.value]
            else:
                target_name = f"{self.value:.2f}"
            text += f"value {target_name}"
        return text


class DecisionTreeClassifier:
    def __init__(self, min_samples: int = 2, max_depth: int = 4, max_splits: int = 25):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.max_splits = max_splits
        self.root = None

    def fit(self, X, y) -> None:
        X = X.astype("float32")
        y = y.astype("int16")
        mask = np.ones_like(y, dtype=bool)
        n_classes = jnp.size(jnp.bincount(y))
        self.root = TreeNode(
            X,
            y,
            mask,
            min_samples=self.min_samples,
            depth=self.max_depth,
            max_splits=self.max_splits,
            n_classes=n_classes,
        )

    def predict(self, X) -> jnp.DeviceArray:
        X = X.astype("float32")
        mask = np.ones((X.shape[0],), dtype=bool)
        if self.root is None:
            raise ValueError("The model is not fitted.")
        return self.root.predict(X, mask).astype("int16")

    def score(self, X: jnp.DeviceArray, y: jnp.DeviceArray) -> float:
        preds = self.predict(X)
        return jnp.mean(preds == y)
