import typing as t

import numpy as np
import pygraphviz as pgv

N_SPLITS = 30

T = t.TypeVar("T")


def entropy(target: np.ndarray) -> float:
    """Shannon entropy in bits."""
    _, classes_counts = np.unique(target, return_counts=True)
    probs = classes_counts / len(target)
    return -np.sum(probs * np.log2(probs))


def most_frequent(arr: np.ndarray) -> t.Any:
    """Return the most frequent element of an array."""
    uniques, counts = np.unique(arr, return_counts=True)
    return int(uniques[np.argmax(counts)])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score for regressors."""
    u = np.square(y_true - y_pred).sum()
    v = np.square(y_true - np.mean(y_true)).sum()
    return 1 - u / v


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score for classifiers."""
    return np.mean(y_true == y_pred)


CRITERIONS = {
    "entropy": entropy,
    "variance": np.var,
}


def split_points(arr: np.ndarray) -> np.ndarray:
    """Return the possible split points to consider.

    For categoricals, we simply return the category list. For continuous
    variables we compute the quantiles.
    """
    uniques = np.unique(arr)
    if len(uniques) <= N_SPLITS:
        return uniques
    else:
        delta = 1 / (N_SPLITS + 1)
        quantiles = np.quantile(arr, np.linspace(delta, 1 - delta, N_SPLITS))
        return np.unique(quantiles)


class Node:
    """Base class for the DecisionTree element.

    This class holds some data and decides whether to split its own data into
    smaller nodes.
    """

    def __init__(
        self,
        data: np.ndarray,
        max_depth: int,
        min_samples: int,
        criterion: t.Callable,
        criterion_name: str,
        feature_names: t.Optional[t.List[str]] = None,
        target_names: t.Optional[t.List[str]] = None,
        uuid: str = "",
        node_type: str = "classifier",
    ) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.criterion = criterion
        self.criterion_name = criterion_name
        self.feature_names = feature_names
        self.target_names = target_names
        self.uuid = uuid
        self.node_type = node_type

        n_samples, _ = data.shape
        self.n_samples = n_samples
        self.score = self.criterion(data)

        # Don't split further if no more depth
        # Stop here if not enough samples for further splits
        if max_depth > 0 and n_samples > self.min_samples:
            self.split_node(data)
        else:
            self.set_leaf(data)

    def set_leaf(self, data: np.ndarray) -> None:
        self.is_leaf = True
        target = data[:, -1]
        if self.node_type == "classifier":
            self.leaf_value = most_frequent(target)
        else:
            self.leaf_value = np.mean(target)

    def split_node(self, data: np.ndarray) -> None:
        """Split the node's data into smaller nodes."""
        n_samples, n_cols = data.shape

        self.best_col = None
        best_score = np.inf
        for col in range(n_cols - 1):
            points = split_points(data[:, col])
            for point in points:
                mask = data[:, col] >= point
                left, right = data[mask, :], data[~mask, :]
                n_left, n_right = left.shape[0], right.shape[0]
                if n_left < self.min_samples or n_right < self.min_samples:
                    continue

                left_score = self.criterion(left[:, -1])
                right_score = self.criterion(right[:, -1])
                score = (n_left * left_score + n_right * right_score) / n_samples

                if score < best_score:
                    best_score = score
                    self.best_col = col
                    self.best_point = point
                    self.best_score = score

        if not self.best_col:
            self.set_leaf(data)
            return
        else:
            self.is_leaf = False

        mask = data[:, self.best_col] >= self.best_point
        left, right = data[mask, :], data[~mask, :]

        self.left_node = Node(
            left,
            self.max_depth - 1,
            self.min_samples,
            self.criterion,
            self.criterion_name,
            self.feature_names,
            self.target_names,
            self.uuid + "l",
            self.node_type,
        )
        self.right_node = Node(
            right,
            self.max_depth - 1,
            self.min_samples,
            self.criterion,
            self.criterion_name,
            self.feature_names,
            self.target_names,
            self.uuid + "r",
            self.node_type,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Recursively predict until a leaf node is reached."""
        if self.is_leaf:
            return np.array([self.leaf_value] * X.shape[0])
        else:
            mask = X[:, self.best_col] >= self.best_point
            left, right = X[mask, :], X[~mask, :]
            left_pred = self.left_node.predict(left)
            right_pred = self.right_node.predict(right)
            pred = np.zeros((X.shape[0]))
            pred[mask] = left_pred
            pred[~mask] = right_pred
            return pred

    def __repr__(self) -> str:
        text = f"n={self.n_samples}\n"
        text += f"{self.criterion_name} {self.score:.2f}\n"
        if not self.is_leaf:
            if self.feature_names is not None:
                col_name = self.feature_names[self.best_col]
            else:
                col_name = f"feature {self.best_col}"
            text += f"{col_name} >= {self.best_point:.2f}"
        else:
            if self.target_names is not None:
                target_name = self.target_names[self.leaf_value]
            else:
                target_name = f"{self.leaf_value:.2f}"
            text += f"value {target_name}"
        return text

    def accept(self, graph: pgv.AGraph) -> None:
        """Visitor to build the graphviz representation."""
        if not self.is_leaf:
            graph.add_node(self.uuid, label=str(self))
            graph.add_node(self.right_node.uuid, label=str(self.right_node))
            graph.add_node(self.left_node.uuid, label=str(self.left_node))

            graph.add_edge(self.uuid, self.right_node.uuid, label="no")
            graph.add_edge(self.uuid, self.left_node.uuid, label="yes")
            self.left_node.accept(graph)
            self.right_node.accept(graph)


class DecisionTree:
    def __init__(
        self,
        criterion: str,
        max_depth: int = 3,
        min_samples: int = 1,
        feature_names: t.Optional[t.List[str]] = None,
        target_names: t.Optional[t.List[str]] = None,
        type: str = "classifier",
    ) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.criterion = CRITERIONS[criterion]
        self.criterion_name = criterion
        self.feature_names = feature_names
        self.target_names = target_names
        self.node_type = type

        if min_samples < 1:
            raise ValueError("`min_samples` must be greater or equal to 1.")

    def dot(self) -> str:
        G = pgv.AGraph(directed=True)
        G.node_attr.update(shape="box")
        self.root.accept(G)
        return G.string()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        data = np.concatenate([X, np.expand_dims(y, axis=1)], axis=1)
        self.root = Node(
            data,
            self.max_depth,
            self.min_samples,
            self.criterion,
            self.criterion_name,
            self.feature_names,
            self.target_names,
            "r",
            self.node_type,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.root.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError


class DecisionTreeClassifier(DecisionTree):
    def __init__(
        self,
        max_depth: int = 3,
        min_samples: int = 1,
        feature_names: t.Optional[t.List[str]] = None,
        target_names: t.Optional[t.List[str]] = None,
    ) -> None:
        super().__init__(
            criterion="entropy",
            max_depth=max_depth,
            min_samples=min_samples,
            feature_names=feature_names,
            target_names=target_names,
            type="classifier",
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return accuracy(y, preds)


class DecisionTreeRegressor(DecisionTree):
    def __init__(
        self,
        max_depth: int = 3,
        min_samples: int = 1,
        feature_names: t.Optional[t.List[str]] = None,
        target_names: t.Optional[t.List[str]] = None,
    ) -> None:
        super().__init__(
            criterion="variance",
            max_depth=max_depth,
            min_samples=min_samples,
            feature_names=feature_names,
            target_names=target_names,
            type="regressor",
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return r2_score(y, preds)
