import jax.numpy as jnp
import numpy as np
import pytest

from jax_trees.jax.utils import row_to_nan, split_points

N_SAMPLES = 100


@pytest.fixture
def mask():
    mask = np.ones((N_SAMPLES,), dtype=bool)
    mask[0] = False
    return mask


@pytest.fixture
def X():
    X = np.stack(
        [
            np.random.randint(low=-10, high=0, size=(N_SAMPLES,)),
            np.random.randint(low=1, high=10, size=(N_SAMPLES,)),
        ],
        axis=1,
    )
    X = X.astype("float")
    return X


def test_row_to_nan(X, mask):
    masked_X = row_to_nan(X, mask)
    assert jnp.all(jnp.isnan(masked_X[0, :]))
    assert jnp.all(masked_X[1:, :] == X[1:, :])


@pytest.mark.parametrize("max_splits", [5, 10, 20])
def test_split_points(X, mask, max_splits):
    max_splits
    points = split_points(X, mask, max_splits)
    assert points.shape == (max_splits, X.shape[1])
