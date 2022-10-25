from decision_tree.numpy.decision_tree import entropy


def test_homogeneity():
    h = entropy([1] * 10)
    assert h == 0


def test_uniform():
    h = entropy([0, 1])
    assert h == 1.0


def test_uniform_2():
    h = entropy([0, 1, 2])
    assert h > 1.5
