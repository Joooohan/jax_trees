from functools import partial
from typing import Tuple

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
