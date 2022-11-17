import jax
import jax.numpy as jnp

from ..regressor import DecisionTreeRegressor, r2_score


class RandomForestRegressor:
    def __init__(
        self,
        n_estimators: int = 100,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
    ):
        self.n_estimators = n_estimators
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.max_splits = max_splits
        self.estimators = None

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        X = X.astype("float32")
        y = y.astype("float32")

        self.estimators = []
        n_samples = X.shape[0]
        key = jax.random.PRNGKey(seed=0)
        for _ in range(self.n_estimators):
            key, subkey = jax.random.split(key)
            model = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                max_splits=self.max_splits,
            )
            idx = jax.random.randint(
                subkey, shape=(n_samples,), minval=0, maxval=n_samples
            )
            model.fit(X, y, jnp.bincount(idx, length=n_samples))
            self.estimators.append(model)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.estimators is None:
            raise ValueError("The model is not fitted.")

        X = X.astype("float32")
        preds = jnp.stack(
            [estimator.predict(X) for estimator in self.estimators],
            axis=0,
        )
        return jnp.mean(preds, axis=0)

    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        preds = self.predict(X)
        return r2_score(y, preds)
