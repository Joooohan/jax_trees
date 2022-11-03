import jax.numpy as jnp
import numpy as np

from decision_tree.jax.regressor import variance


def test_variance():
    y = np.asarray([1.0, 1.0, 0.0, 0.0])
    mask = np.asarray([True, True, True, True])
    score = variance(y, mask)
    assert score == 0.25

    mask = np.asarray([False, False, True, True])
    score = variance(y, mask)
    assert score == 0.0


def test_variance_empty():
    y = np.asarray([1.0, 1.0])
    mask = np.asarray([0, 0]).astype("bool")
    score = variance(y, mask)
    assert jnp.isnan(score)
