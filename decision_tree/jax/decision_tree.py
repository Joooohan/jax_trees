from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap


@partial(vmap, in_axes=(1, None), out_axes=1)
def row_to_nan(X, mask):
    return jnp.where(mask, X, np.nan)


@partial(jit, static_argnames="max_splits")
def split_points(X, mask, max_splits: int):
    """Generate split points for the X data."""
    X = row_to_nan(X, mask)

    batched_unique = vmap(
        partial(jnp.unique, size=max_splits, fill_value=np.nan), in_axes=1, out_axes=1
    )
    delta = 1 / (max_splits + 1)
    quantiles = jnp.nanquantile(X, jnp.linspace(delta, 1 - delta, max_splits), axis=0)
    uniques = batched_unique(X)
    candidates = jnp.concatenate([uniques, quantiles], axis=0)
    points = batched_unique(candidates)
    return points


@partial(jit, static_argnames=["n_classes"])
def entropy(y, mask, n_classes):
    n_samples = y.shape[0]
    counts = jnp.bincount(jnp.where(mask, y, n_classes), length=n_classes)
    probs = counts / n_samples
    return -jnp.sum(probs * jnp.log2(probs))


@partial(jit, static_argnames=["n_classes"])
@partial(vmap, in_axes=(1, None, None, 1, None), out_axes=1)
@partial(vmap, in_axes=(None, None, None, 0, None))
def compute_scores(X_col, y, mask, split_value, n_classes):
    left_mask = jnp.where(X_col >= split_value, mask, False)
    right_mask = jnp.where(X_col < split_value, mask, False)

    left_score = entropy(y, left_mask, n_classes)
    right_score = entropy(y, right_mask, n_classes)

    n_left = jnp.sum(left_mask)
    n_right = jnp.sum(right_mask)

    avg_score = (n_left * left_score + n_right * right_score) / (n_left + n_right)

    return avg_score


@partial(jit, static_argnames=["max_splits", "n_classes"])
def split_node(X, y, mask, max_splits: int, n_classes: int):
    """The algorithm does the following:

    1. Generate split points (N_SPLITS, N_COLS) matrix
    2. For each split point, compute the split score (N_SPLITS, N_COLS)
    3. Select the point with the lowest score
    4. Generate two new masks for left and right children
    """
    points = split_points(X, mask, max_splits)
    scores = compute_scores(X, y, mask, points, n_classes)

    split_row, split_col = jnp.unravel_index(jnp.nanargmin(scores), scores.shape)
    split_value = scores[split_row, split_col]

    left_mask = jnp.where(X[:, split_col] >= split_value, mask, False)
    right_mask = jnp.where(X[:, split_col] < split_value, mask, False)
    return left_mask, right_mask


class DecisionTreeClassifier:
    def __init__(self):
        ...

    def fit(X, y) -> None:
        ...
