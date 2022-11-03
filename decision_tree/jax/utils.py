from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import jit, vmap


@partial(vmap, in_axes=(1, None), out_axes=1)
def row_to_nan(X, mask):
    """Convert a whole row of X to nan with a row mask."""
    return jnp.where(mask, X, np.nan)


@partial(jit, static_argnames="max_splits")
def split_points(X, mask, max_splits: int):
    """Generate split points for the data."""
    X = row_to_nan(X, mask)
    delta = 1 / (max_splits + 1)
    quantiles = jnp.nanquantile(X, jnp.linspace(delta, 1 - delta, max_splits), axis=0)
    return quantiles
