import numpy as np

from ..decision_tree import entropy


def test_homogeneity():
    h = entropy([1] * 10)
    assert h == 0
