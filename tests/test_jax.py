import jax.numpy as jnp
import numpy as np
import pytest
from decision_tree.jax.decision_tree import (
    compute_all_scores,
    compute_column_scores,
    compute_score,
    entropy,
    most_frequent,
    row_to_nan,
    split_node,
    split_points,
)

N_SAMPLES = 100
N_CLASSES = 3


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


@pytest.fixture
def y():
    y = np.random.randint(low=0, high=N_CLASSES, size=(N_SAMPLES,))
    y = y.astype("int")
    return y


@pytest.fixture
def mask():
    mask = np.ones((N_SAMPLES,), dtype=bool)
    mask[0] = False
    return mask


def test_row_to_nan(X, mask):
    masked_X = row_to_nan(X, mask)
    assert jnp.all(jnp.isnan(masked_X[0, :]))
    assert jnp.all(masked_X[1:, :] == X[1:, :])


@pytest.mark.parametrize("max_splits", [5, 10, 20])
def test_split_points(X, mask, max_splits):
    max_splits
    points = split_points(X, mask, max_splits)
    assert points.shape == (max_splits, X.shape[1])


def test_entropy():
    y = np.asarray([1, 1, 0, 0])
    mask = np.asarray([True, True, True, True])
    score = entropy(y, mask, n_classes=2)
    assert score == 1.0

    mask = np.asarray([False, False, True, True])
    score = entropy(y, mask, n_classes=2)
    assert score == 0.0


def test_entropy_empty():
    y = np.asarray([1, 1])
    mask = np.asarray([0, 0]).astype("bool")
    score = entropy(y, mask, n_classes=2)
    assert jnp.isnan(score)


def test_compute_score():
    X = np.asarray([1, 2, 3, 4]).astype("float")
    y = np.asarray([0, 0, 1, 1])
    mask = np.asarray([1, 1, 1, 1]).astype("bool")

    split_value = 2.5
    score = compute_score(X, y, mask, split_value, n_classes=2)
    assert score == 0.0

    split_value = 1.5
    mask = np.asarray([1, 1, 0, 1]).astype("bool")
    score = compute_score(X, y, mask, split_value, n_classes=2)
    assert score == 2 / 3

    split_value = 5
    mask = np.asarray([1, 1, 1, 1]).astype("bool")
    score = compute_score(X, y, mask, split_value, n_classes=2)
    assert jnp.isnan(score)

    split_value = -3
    mask = np.asarray([1, 1, 1, 1]).astype("bool")
    score = compute_score(X, y, mask, split_value, n_classes=2)
    assert jnp.isnan(score)


def test_compute_column_scores():
    X = np.asarray([1, 2, 3, 4]).astype("float")
    y = np.asarray([0, 0, 1, 1])
    mask = np.asarray([1, 1, 1, 1]).astype("bool")
    split_values = np.asarray([2.5, 1.5, 5, -4]).astype("float")
    scores = compute_column_scores(X, y, mask, split_values, 2)
    assert jnp.allclose(
        scores, np.asarray([0.0, 2 / 3, np.nan, np.nan]), equal_nan=True, rtol=1e-1
    )


def test_compute_all_scores():
    X = np.stack([[1, 2, 3, 4], [1, 0, 1, 0]], axis=1).astype("float")
    y = np.asarray([0, 0, 1, 1])
    mask = np.asarray([1, 1, 1, 1]).astype("bool")
    split_values = np.stack([[2.5, -1], [4, 0.5]], axis=1).astype("float")
    scores = compute_all_scores(X, y, mask, split_values, 2)
    assert jnp.allclose(
        scores, np.stack([[0.0, np.nan], [np.nan, 1.0]], axis=1), equal_nan=True
    )


@pytest.mark.parametrize("max_splits", [5, 10, 20])
def test_split_node(X, y, mask, max_splits):
    left_mask, right_mask, _, _ = split_node(X, y, mask, max_splits, N_CLASSES)
    assert jnp.sum(left_mask) > 0
    assert jnp.sum(right_mask) > 0
    assert jnp.sum(left_mask) + jnp.sum(right_mask) == jnp.sum(mask)


def test_most_frequent():
    y = np.asarray([0, 1, 0, 1, 0])
    mask = np.asarray([1, 1, 1, 1, 1]).astype("bool")
    value = most_frequent(y, mask, 2)
    assert value == 0

    mask = np.asarray([0, 1, 0, 1, 1]).astype("bool")
    value = most_frequent(y, mask, 2)
    assert value == 1
