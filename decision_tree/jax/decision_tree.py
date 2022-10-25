from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import jit, vmap


def split_node(X, y, mask):
    """The algorithm does the following:

    1. Generate split points (N_SPLITS, N_COLS) matrix
    2. For each split point, compute the split score (N_SPLITS, N_COLS)
    3. Select the point with the lowest score
    4. Generate two new masks for left and right children
    """


@partial(jit, static_argnames="max_splits")
def split_points(X, mask, max_splits: int):
    """Generate split points for the X data."""
    X = jnp.where(mask, X, np.nan)

    batched_unique = vmap(
        partial(jnp.unique, size=max_splits, fill_value=np.nan), in_axes=1, out_axes=1
    )
    delta = 1 / (max_splits + 1)
    quantiles = jnp.nanquantile(X, jnp.linspace(delta, 1 - delta, max_splits), axis=0)
    uniques = batched_unique(X)
    candidates = jnp.concatenate([uniques, quantiles], axis=0)
    points = batched_unique(candidates)
    return points


class DecisionTreeClassifier:
    def __init__(self):
        ...

    def fit(X, y) -> None:
        ...
