import jax.numpy as jnp
import numpy as np


def variance(y, mask):
    """Masked version of variance."""
    return jnp.nanvar(jnp.where(mask, y, np.nan))


class DecisionTreeRegressor:
    def __init__(self, min_samples: int = 2, max_depth: int = 4, max_splits: int = 25):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.max_splits = max_splits
        self.root = None

    def fit(self, X, y) -> None:
        X = X.astype("float")
        y = y.astype("float")
        mask = np.ones_like(y, dtype=bool)
        n_classes = jnp.size(jnp.bincount(y))
        self.root = TreeNode(
            X,
            y,
            mask,
            min_samples=self.min_samples,
            depth=self.max_depth,
            max_splits=self.max_splits,
            n_classes=n_classes,
        )

    def predict(self, X) -> jnp.DeviceArray:
        X = X.astype("float")
        mask = np.ones((X.shape[0],), dtype=bool)
        if self.root is None:
            raise ValueError("The model is not fitted.")
        return self.root.predict(X, mask).astype("float")

    def score(self, X: jnp.DeviceArray, y: jnp.DeviceArray) -> float:
        preds = self.predict(X)
        return jnp.mean(preds == y)
