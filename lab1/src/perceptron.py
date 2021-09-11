from typing import Tuple

import numpy as np
import pandas as pd


def _fit_delta_online(weights: np.array, X: np.array, y: np.array, learning_rate: float, classes: Tuple) -> np.array:
    for index in range(len(X)):
        weights = weights - learning_rate * ((weights @ X[:, index] - y[:, index]) * X[:, index].T)

    return weights


def _fit_delta_batch(weights: np.array, X: np.array, y: np.array, learning_rate: float, classes: Tuple) -> np.array:
    return weights - learning_rate * ((weights @ X - y) @ X.T)


def _fit_perceptron(weights: np.array, X: np.array, y: np.array, learning_rate: float, classes: Tuple) -> np.array:
    for index in range(len(X)):
        prediction = np.where(weights @ X[:, index] > 0, classes[0], classes[1]).item()
        ground_truth = y[:, index].item()

        if prediction != ground_truth:
            weights = weights + learning_rate * X[:, index] if prediction == classes[1] \
                else weights - learning_rate * X[:, index]

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

    def __init__(
            self,
            fit_intercept: bool = True,
            learning_rule: str = 'delta_batch',
            learning_rate: float = 1e-3,
            max_iterations: int = 100,
            tolerance: float = None,
            warm_start: bool = False,
            classes: Tuple = (1, -1)
    ):
        self.fit_intercept = fit_intercept
        self.learning_rule = learning_rule
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.warm_start = warm_start
        self.classes = classes

    def fit(self, X, y) -> None:
        if X is pd.DataFrame: X = X.to_numpy()
        if y is pd.DataFrame: y = y.to_numpy()

        if self.fit_intercept:
            X = np.hstack(
                (np.reshape(np.ones(len(X)), (len(X), 1)),
                 X)
            )

        X = X.T
        y = y.T

        weights = self.weights if self.warm_start \
            else np.random.normal(size=(1, X.shape[0]))

        learning_rule = learning_rules[self.learning_rule]

        for epoch in range(self.max_iterations):  # apply learning rule to weights and if converged break
            weights = learning_rule(weights, X, y, self.learning_rate, self.classes)

        weights = weights.flatten()
        self.weights = weights

    @property
    def intercept_(self):
        return self.weights[0] if self.fit_intercept else 0

    @property
    def coef_(self):
        return self.weights[1:] if self.fit_intercept else self.weights

    def predict(self, X) -> np.array:
        result = self.coef_ @ X.T + self.intercept_

        return np.where(result > 0, self.classes[0], self.classes[1])
