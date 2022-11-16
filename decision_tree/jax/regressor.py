from functools import partial
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import jit, vmap

from .utils import split_mask, split_points


def variance(y: jnp.ndarray, mask: jnp.ndarray) -> float:
    """Weighted version of variance."""
    avg = jnp.average(y, weights=mask)
    return jnp.average(jnp.square(y - avg), weights=mask)


def average(y: jnp.ndarray, mask: jnp.ndarray) -> float:
    """Masked version of the mean."""
    return jnp.average(y, weights=mask)


def r2_score(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Score for regressors."""
    u = jnp.square(y_true - y_pred).sum()
    v = jnp.sum(jnp.square(y_true - jnp.mean(y_true)))
    return 1 - u / v


def compute_score(
    X_col: jnp.ndarray, y: jnp.ndarray, mask: jnp.ndarray, split_value: float
) -> float:
    """Compute the scores of data splits."""
    left_mask, right_mask = split_mask(split_value, X_col, mask)

    left_score = variance(y, left_mask)
    right_score = variance(y, right_mask)

    n_left = jnp.sum(left_mask)
    n_right = jnp.sum(right_mask)

    avg_score = (n_left * left_score + n_right * right_score) / (
        n_left + n_right
    )

    return avg_score


compute_column_scores = vmap(compute_score, in_axes=(None, None, None, 0))

compute_all_scores = vmap(
    compute_column_scores,
    in_axes=(1, None, None, 1),
    out_axes=1,
)


@partial(jit, static_argnames=["max_splits"])
def split_node(
    X: jnp.ndarray,
    y: jnp.ndarray,
    mask: jnp.ndarray,
    max_splits: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, int]:
    """The algorithm does the following:

    1. Generate split points candidates (N_SPLITS, N_COLS) matrix
    2. For each split point, compute the split score -> (N_SPLITS, N_COLS)
    3. Select the point with the lowest score
    4. Generate two new masks for left and right children nodes
    """
    points = split_points(X, mask, max_splits)
    scores = compute_all_scores(X, y, mask, points)

    split_row, split_col = jnp.unravel_index(
        jnp.nanargmin(scores), scores.shape
    )
    split_value = points[split_row, split_col]
    left_mask, right_mask = split_mask(split_value, X[:, split_col], mask)
    return left_mask, right_mask, split_value, split_col


class TreeNode:
    def __init__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        mask: jnp.ndarray,
        min_samples: int,
        depth: int,
        max_splits: int,
    ):
        self.n_samples = jnp.sum(mask)
        self.score = variance(y, mask)
        self.feature_names = None
        self.target_names = None

        if jnp.sum(mask) > min_samples and depth > 0:
            left_mask, right_mask, split_value, split_col = split_node(
                X, y, mask, max_splits
            )
            self.is_leaf = False
            self.left_node = TreeNode(
                X, y, left_mask, min_samples, depth - 1, max_splits
            )
            self.right_node = TreeNode(
                X, y, right_mask, min_samples, depth - 1, max_splits
            )
            self.split_value = split_value
            self.split_col = split_col
        else:
            self.is_leaf = True
            self.value = average(y, mask)

    def predict(self, X: jnp.ndarray, mask: jnp.ndarray) -> jnp.DeviceArray:
        if self.is_leaf:
            return jnp.where(mask, self.value, jnp.nan)
        else:
            left_mask = jnp.where(
                X[:, self.split_col] >= self.split_value, mask, False
            )
            right_mask = jnp.where(
                X[:, self.split_col] < self.split_value, mask, False
            )
            right_pred = self.right_node.predict(X, right_mask)
            left_pred = self.left_node.predict(X, left_mask)
            return jnp.where(
                left_mask,
                left_pred,
                jnp.where(right_mask, right_pred, jnp.nan),
            )

    def __repr__(self) -> str:
        text = f"n={self.n_samples}\n"
        text += f"variance {self.score:.2f}\n"
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


class DecisionTreeRegressor:
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
        y = y.astype("float32")
        if mask is None:
            mask = jnp.ones_like(y)
        self.root = TreeNode(
            X,
            y,
            mask,
            min_samples=self.min_samples,
            depth=self.max_depth,
            max_splits=self.max_splits,
        )

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        X = X.astype("float32")
        mask = jnp.ones((X.shape[0],))
        if self.root is None:
            raise ValueError("The model is not fitted.")
        return self.root.predict(X, mask).astype("float32")

    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        preds = self.predict(X)
        return r2_score(y, preds)
