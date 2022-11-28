from functools import partial
from typing import Callable, Tuple

import jax.numpy as jnp
from jax import jit, vmap


@partial(vmap, in_axes=(1, None), out_axes=1)
def row_to_nan(X: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Convert a whole row of X to nan with a row mask."""
    return jnp.where(mask > 0, X, jnp.nan)


@partial(jit, static_argnames="max_splits")
def split_points(
    X: jnp.ndarray, mask: jnp.ndarray, max_splits: int
) -> jnp.ndarray:
    """Generate split points for the data."""
    X = row_to_nan(X, mask)
    delta = 1 / (max_splits + 1)
    quantiles = jnp.nanquantile(
        X, jnp.linspace(delta, 1 - delta, max_splits), axis=0
    )
    return quantiles


def split_mask(
    value: float, col: jnp.ndarray, mask: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    left_mask = jnp.where(col >= value, mask, 0)
    right_mask = jnp.where(col < value, mask, 0)
    return left_mask, right_mask


def compute_score_generic(
    X_col: jnp.ndarray,
    y: jnp.ndarray,
    mask: jnp.ndarray,
    split_value: float,
    score_fn: Callable,
) -> float:
    """Compute the scores of data splits."""
    left_mask, right_mask = split_mask(split_value, X_col, mask)

    left_score = score_fn(y, left_mask)
    right_score = score_fn(y, right_mask)

    n_left = jnp.sum(left_mask)
    n_right = jnp.sum(right_mask)

    avg_score = (n_left * left_score + n_right * right_score) / (
        n_left + n_right
    )

    return avg_score


def make_scoring_function(score_fn: Callable) -> Callable:
    compute_score_specialized = partial(
        compute_score_generic, score_fn=score_fn
    )
    compute_column_scores = vmap(
        compute_score_specialized, in_axes=(None, None, None, 0)
    )

    compute_all_scores = vmap(
        compute_column_scores,
        in_axes=(1, None, None, 1),
        out_axes=1,
    )
    return compute_all_scores


def split_node_generic(
    X: jnp.ndarray,
    y: jnp.ndarray,
    mask: jnp.ndarray,
    max_splits: int,
    compute_all_scores: Callable,
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


def make_split_node_function(score_fn: Callable) -> Callable:
    compute_all_scores = make_scoring_function(score_fn)
    split_node_specialized = partial(
        split_node_generic, compute_all_scores=compute_all_scores
    )
    return split_node_specialized
