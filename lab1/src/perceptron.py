import numpy as np


class Perceptron:
    coef_: np.array
    intercept_: float
    fit_intercept: bool
    learning_rule: str
    mode: str
    learning_rate: float
    max_iterations: int
    tolerance: float

    def __init__(
            self,
            fit_intercept: bool = True,
            learning_rule: str = 'delta',
            mode: str = 'batch',
            learning_rate: float = 1e-1,
            max_iterations: int = 100,
            tolerance: float = None
    ):
        self.fit_intercept = fit_intercept
        self.learning_rule = learning_rule
        self.mode = mode
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def fit(self, X: np.array, y: np.array) -> None:
        if self.fit_intercept:
            X = np.hstack(np.ones(len(X)), X)

        weights = np.random.normal(size=len(X))

        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

    def predict(self, X: np.array) -> np.array:
        return self.coef_ @ X + self.intercept_
