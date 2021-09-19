from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.gaussian_process.kernels import RBF


def _fit_delta_online(
        weights: np.array,
        X: np.array,
        y: np.array,
        learning_rate: float,
        max_iterations: int,
        error_per_epoch: dict
) -> np.array:
    for epoch in range(max_iterations):
        for index in range(len(X.T)):
            weights = weights - learning_rate * ((weights @ X[:, index] - y[:, index]) * X[:, index].T)

        raw_prediction = weights @ X
        error_per_epoch['mse'].append(mean_squared_error(y, raw_prediction))

    return weights


def _fit_delta_batch(
        weights: np.array,
        X: np.array,
        y: np.array,
        learning_rate: float,
        max_iterations: int,
        error_per_epoch: dict
) -> np.array:
    for epoch in range(max_iterations):
        weights = weights - learning_rate * ((weights @ X - y) @ X.T)

        raw_prediction = weights @ X
        error_per_epoch['mse'].append(mean_squared_error(y, raw_prediction))

    return weights


def _fit_least_squares(
        weights: np.array,
        X: np.array,
        y: np.array,
        learning_rate: float,
        max_iterations: int,
        error_per_epoch: dict
) -> np.array:
    weights = np.linalg.inv(X @ X.T) @ X @ y.T
    weights = weights.T

    raw_prediction = weights @ X
    error_per_epoch['mse'] = [mean_squared_error(y, raw_prediction) for _ in range(max_iterations)]

    return weights


def _map_to_rbf(
        X: np.array,
        rbf_locations: np.array,
        rbf_scale: float
):
    return RBF(length_scale=rbf_scale).__call__(X=X, Y=rbf_locations)


def _competitive_learning(X: np.array, learning_rate: float, max_iterations: int, nodes: int) -> np.array:
    rbf_locations = np.random.normal(size=(nodes, X.shape[1]))

    for _ in range(max_iterations):
        for x in X:
            diff = (rbf_locations - x)
            distances = diff @ diff.T
            winner_index = np.argmax(np.diag(distances))

            rbf_locations[winner_index] = rbf_locations[winner_index] \
                                          + learning_rate * (x - rbf_locations[winner_index])

    return rbf_locations


learning_rules = {
    'delta_batch': _fit_delta_batch,
    'delta_online': _fit_delta_online,
    'least_squares': _fit_least_squares
}


class RBFNetwork:
    coef_: np.array
    fit_intercept: bool
    learning_rule: str
    learning_rate: float
    max_iterations: int
    tolerance: float
    warm_start: bool
    classes: Tuple
    weights: np.array
    error_per_epoch: dict
    weight_init_loc: int
    weight_init_scale: int
    rbf_locations: np.array
    rbf_scale: float

    def __init__(
            self,
            rbf_locations: np.array,
            rbf_scale: float = 1,
            learning_rule: str = 'delta_batch',
            learning_rate: float = 1e-3,
            max_iterations: int = 100,
            weight_init_loc: int = 0,
            weight_init_scale: int = 1
    ):
        self.learning_rule = learning_rule
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.error_per_epoch = {
            'mse': []
        }
        self.weight_init_loc = weight_init_loc
        self.weight_init_scale = weight_init_scale
        self.rbf_locations = rbf_locations
        self.rbf_scale = rbf_scale

    def fit(self, X, y) -> None:
        if X is pd.DataFrame: X = X.to_numpy()
        if y is pd.DataFrame: y = y.to_numpy()

        self.error_per_epoch = {
            'mse': []
        }

        X = _map_to_rbf(X=X, rbf_locations=self.rbf_locations, rbf_scale=self.rbf_scale).T
        y = y.T

        self.weights = np.random.normal(size=(1, X.shape[0]), loc=self.weight_init_loc, scale=self.weight_init_scale)

        self.weights = learning_rules[self.learning_rule](
            self.weights,
            X,
            y,
            self.learning_rate,
            self.max_iterations,
            self.error_per_epoch
        )

    @property
    def intercept_(self):
        return 0

    @property
    def coef_(self):
        return self.weights.flatten()

    def predict(self, X) -> np.array:
        X = _map_to_rbf(X=X, rbf_locations=self.rbf_locations, rbf_scale=self.rbf_scale)

        return self.coef_ @ X.T
