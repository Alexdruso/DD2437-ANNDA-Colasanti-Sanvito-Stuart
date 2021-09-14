from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score


def _fit_delta_online(weights: np.array, X: np.array, y: np.array, learning_rate: float) -> np.array:
    for index in range(len(X.T)):
        weights = weights - learning_rate * ((weights @ X[:, index] - y[:, index]) * X[:, index].T)

    return weights


def _fit_delta_batch(weights: np.array, X: np.array, y: np.array, learning_rate: float) -> np.array:
    return weights - learning_rate * ((weights @ X - y) @ X.T)


def _fit_perceptron(weights: np.array, X: np.array, y: np.array, learning_rate: float) -> np.array:
    for index in range(len(X.T)):
        prediction = np.where(weights @ X[:, index] > 0, 1, -1).item()
        ground_truth = y[:, index].item()
        weights = weights + learning_rate * (ground_truth - prediction) * X[:, index]

    return weights


learning_rules = {
    'delta_batch': _fit_delta_batch,
    'delta_online': _fit_delta_online,
    'perceptron': _fit_perceptron
}


class Perceptron:
    coef_: np.array
    intercept_: float
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

    def __init__(
            self,
            fit_intercept: bool = True,
            learning_rule: str = 'delta_batch',
            learning_rate: float = 1e-3,
            max_iterations: int = 100,
            tolerance: float = None,
            warm_start: bool = False,
            weight_init_loc: int = 0,
            weight_init_scale: int = 1
    ):
        self.fit_intercept = fit_intercept
        self.learning_rule = learning_rule
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.warm_start = warm_start
        self.error_per_epoch = {
            'accuracy': [],
            'mse': []
        }
        self.weight_init_loc = weight_init_loc
        self.weight_init_scale = weight_init_scale

    def fit(self, X, y) -> None:
        if X is pd.DataFrame: X = X.to_numpy()
        if y is pd.DataFrame: y = y.to_numpy()

        self.error_per_epoch = {
            'accuracy': [],
            'mse': []
        }

        if self.fit_intercept:
            X = np.hstack(
                (np.reshape(np.ones(len(X)), (len(X), 1)),
                 X)
            )

        X = X.T
        y = y.T

        self.weights = self.weights if self.warm_start \
            else np.random.normal(size=(1, X.shape[0]), loc=self.weight_init_loc, scale=self.weight_init_scale)

        learning_rule = learning_rules[self.learning_rule]

        for epoch in range(self.max_iterations):  # apply learning rule to weights and if converged break
            self.weights = learning_rule(self.weights, X, y, self.learning_rate)

            raw_prediction = self.weights @ X
            prediction = np.where(raw_prediction > 0, 1, -1)
            self.error_per_epoch['accuracy'].append(accuracy_score(y.flatten(), prediction.flatten()))
            self.error_per_epoch['mse'].append(mean_squared_error(y, raw_prediction))

    @property
    def intercept_(self):
        return self.weights.flatten()[0] if self.fit_intercept else 0

    @property
    def coef_(self):
        return self.weights.flatten()[1:] if self.fit_intercept else self.weights

    def predict(self, X) -> np.array:
        result = self.decision_function(X)

        return np.where(result > 0, 1, -1)

    def decision_function(self, X) -> np.array:
        return self.coef_ @ X.T + self.intercept_
