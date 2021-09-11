import numpy as np
import pandas as pd


def _fit_delta_online(weights: np.array, X: np.array, y: np.array, learning_rate: float) -> np.array:
    for index in range(len(X)):
        weights = weights - 0.1 * ((weights @ X[:, index] - y[:, index]) * X[:, index].T)

    return weights


def _fit_delta_batch(weights: np.array, X: np.array, y: np.array, learning_rate: float) -> np.array:
    return weights - learning_rate * ((weights@X - y) @ X.T)


def _fit_perceptron(weights: np.array, X: np.array, y: np.array, learning_rate: float) -> np.array:
    return None


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

    def __init__(
            self,
            fit_intercept: bool = True,
            learning_rule: str = 'delta_batch',
            learning_rate: float = 1e-1,
            max_iterations: int = 100,
            tolerance: float = None,
            warm_start: bool = False
    ):
        self.fit_intercept = fit_intercept
        self.learning_rule = learning_rule
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.warm_start = warm_start

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

        weights = np.concatenate((self.intercept_, self.coef_)) if self.warm_start \
            else np.random.normal(size=(1, X.shape[0]))

        learning_rule = learning_rules[self.learning_rule]

        for epoch in range(self.max_iterations):  # apply learning rule to weights and if converged break
            weights = learning_rule(weights, X, y, self.learning_rate)

        weights = weights.flatten()

        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

    def predict(self, X) -> np.array:
        return self.coef_ @ X.T + self.intercept_

