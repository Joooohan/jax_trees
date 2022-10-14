import numpy as np
import pygraphviz as pgv

N_SPLITS = 30


def split_points(arr: np.ndarray) -> np.ndarray:
    """Return the possible split points to consider."""
    uniques = np.unique(arr)
    if len(uniques) <= N_SPLITS:
        return uniques
    else:
        delta = 1 / (N_SPLITS + 1)
        quantiles = np.quantile(arr, np.linspace(delta, 1 - delta, N_SPLITS))
        return np.unique(quantiles)


def entropy(target: np.ndarray) -> float:
    """Shannon entropy in bits."""
    _, classes_counts = np.unique(target, return_counts=True)
    probs = classes_counts / len(target)
    return -np.sum(probs * np.log2(probs))


class Node:
    def __init__(
        self, data: np.ndarray, max_depth: int, min_samples: int, uuid: str = ""
    ) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples
        n_samples, _ = data.shape
        self.n_samples = n_samples
        self.score = entropy(data)
        self.uuid = uuid

        # Don't split further if no more depth
        # Stop here if not enough samples for further splits
        if max_depth > 0 and n_samples > self.min_samples:
            self.is_leaf = False
            self.split_node(data)
        else:
            self.is_leaf = True
            target = data[:, -1]
            uniques, counts = np.unique(target, return_counts=True)
            self.class_label = uniques[np.argmax(counts)]

    def split_node(self, data: np.ndarray) -> None:
        n_samples, n_cols = data.shape

        best_score = np.inf
        for col in range(n_cols - 1):
            points = split_points(data[:, col])
            for point in points:
                mask = data[:, col] >= point
                left, right = data[mask, :], data[~mask, :]
                n_left, n_right = left.shape[0], right.shape[0]
                if n_left < self.min_samples or n_right < self.min_samples:
                    continue

                left_score = entropy(left[:, -1])
                right_score = entropy(right[:, -1])
                score = (n_left * left_score + n_right * right_score) / n_samples

                if score < best_score:
                    self.best_col = col
                    self.best_point = point
                    self.best_score = score

        mask = data[:, self.best_col] >= self.best_point
        left, right = data[mask, :], data[~mask, :]

        self.left_node = Node(
            left, self.max_depth - 1, self.min_samples, self.uuid + "l"
        )
        self.right_node = Node(
            right, self.max_depth - 1, self.min_samples, self.uuid + "r"
        )

    def __repr__(self) -> str:
        text = f"n={self.n_samples}\n"
        text += f"entropy {self.score:.2f}\n"
        if not self.is_leaf:
            text += f"col {self.best_col} >= {self.best_point}"
        else:
            text += f"label {self.class_label}"
        return text

    def accept(self, graph: pgv.AGraph) -> None:
        if not self.is_leaf:
            graph.add_node(self.uuid, label=str(self))
            graph.add_node(self.left_node.uuid, label=str(self.left_node))
            graph.add_node(self.right_node.uuid, label=str(self.right_node))

            graph.add_edge(self.uuid, self.left_node.uuid)
            graph.add_edge(self.uuid, self.right_node.uuid)
            self.left_node.accept(graph)
            self.right_node.accept(graph)


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 3, min_samples: int = 1) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples
        if min_samples < 1:
            raise ValueError("`min_samples` must be greater or equal to 1.")

    def dot(self) -> str:
        G = pgv.AGraph(directed=True)
        G.node_attr.update(shape="box")
        self.root.accept(G)
        return G.string()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        data = np.concatenate([X, np.expand_dims(y, axis=1)], axis=1)
        self.root = Node(data, self.max_depth, self.min_samples, "r")

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
