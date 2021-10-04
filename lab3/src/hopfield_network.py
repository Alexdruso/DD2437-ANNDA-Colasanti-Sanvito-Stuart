from typing import List

import numpy as np
from sklearn.base import BaseEstimator


def _sign(X: np.array) -> np.array:
    return np.where(X > 0, 1, -1)


def _predict_batch(weights: np.array, X: np.array, bias: float, sparse: bool) -> np.array:
    return 0.5 + 0.5 * _sign(X @ weights - bias) if sparse else _sign(X @ weights)


def _predict_sequential(weights: np.array, X: np.array, bias: float, sparse: bool) -> np.array:
    features_number = len(X)
    prediction = X.copy()
    for feature in range(0, features_number):
        # N X 1 = N X M @ M X 1
        prediction[:, feature] = _sign(prediction @ weights[:, feature] - bias) if sparse \
            else _sign(prediction @ weights[:, feature])
    return prediction


def _get_energy(weights: np.array, prediction: np.array) -> np.array:
    energy = -prediction @ (weights @ prediction.T)
    return energy


prediction_methods = {
    "batch": _predict_batch,
    "sequential": _predict_sequential
}


class HopfieldNetwork(BaseEstimator):
    weights: np.array
    max_iterations: int
    bias: float
    sparse: bool
    zero_diagonal: bool
    random_weights: bool
    symmetric_weights: bool
    sparsity: float
    prediction_method: str
    energy_per_iteration: List

    def __init__(
            self,
            max_iterations: int = 150,
            bias: float = 0.0,
            zero_diagonal: bool = False,
            random_weights: bool = False,
            symmetric_weights: bool = False,
            sparsity: float = 0.0,
            prediction_method: str = "batch"
    ):
        self.max_iterations = max_iterations
        self.bias = bias
        self.zero_diagonal = zero_diagonal
        self.random_weights = random_weights
        self.symmetric_weights = symmetric_weights
        self.sparsity = sparsity
        self.prediction_method = prediction_method
        self.energy_per_iteration = []

    def fit(self, X: np.array, y: np.array) -> None:
        # X should be shaped like (number of patterns, number of features)
        X = np.array(X)
        features_number = X.shape[1]

        weights = np.random.normal(size=(features_number, features_number)) if self.random_weights \
            else (X - self.sparsity).T @ (X - self.sparsity)

        if self.zero_diagonal:
            np.fill_diagonal(weights, 0)

        if self.symmetric_weights:
            weights = 0.5 * (weights + weights.T)

        self.weights = weights / features_number

    def predict(self, X: np.array) -> np.array:
        prediction = X

        prediction_method = prediction_methods[self.prediction_method]

        for _ in range(self.max_iterations):
            prediction = prediction_method(self.weights, prediction, self.bias, self.sparse)
            self.energy_per_iteration.append(_get_energy(self.weights, prediction))

        return prediction
