import numpy as np
from typing import Tuple


class TwoLayerPerceptron:
    mode: str
    learning_rate: float
    momentum: float
    max_iterations: int
    tolerance: float
    hidden_layer_size: int
    validation_fraction: float

    classes_: np.array
    n_outputs_: int
    coefs_: list
    intercepts_: list
    prev_delta_W_: float
    prev_delta_V_: float
    loss_curve_: np.array
    val_loss_curve_: np.array

    def __init__(
            self,
            mode: str = 'batch',
            learning_rate: float = 1e-3,
            momentum: float = 0.9,
            max_iterations: int = 100,
            tolerance: float = None,
            hidden_layer_size: int = None,
            validation_fraction: float = 0.2,
    ):
        self.mode = mode
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.hidden_layer_size = hidden_layer_size
        self.validation_fraction = validation_fraction

    def _transfer_function(self, x: np.array) -> np.array:
        return 2/(1 + np.exp(-x)) - 1

    def _transfer_function_derivative(self, calculated_transfer_function: np.array) -> np.array:
        return np.multiply((1 + calculated_transfer_function), (1 - calculated_transfer_function))/2

    def _forward_pass(self, X: np.array, W: np.array, V: np.array) -> Tuple[np.array]:
        H = self._transfer_function(W @ X)
        H = np.pad(H, pad_width=((0, 1), (0, 0)),
                   mode='constant', constant_values=1)
        O = self._transfer_function(V @ H)
        return H, O

    def _backward_pass(self, X: np.array, y: np.array, H: np.array, O: np.array, V: np.array) -> Tuple[np.array]:
        delta_O = np.multiply((O - y), self._transfer_function_derivative(O))
        delta_H = np.multiply(V.transpose() @ delta_O,
                              self._transfer_function_derivative(H))
        delta_H = delta_H[:-1, :]
        return delta_H, delta_O

    def _weight_update(self, X: np.array, W: np.array, V: np.array, H: np.array, delta_H: np.array, delta_O: np.array) -> Tuple[np.array]:
        delta_W = self.momentum * self.prev_delta_W_ - \
            (1 - self.momentum) * (delta_H @ X.transpose())
        delta_V = self.momentum * self.prev_delta_V_ - \
            (1 - self.momentum) * (delta_O @ H.transpose())

        self.prev_delta_W_ = delta_W
        self.prev_delta_V_ = delta_V

        new_W = W + self.learning_rate * delta_W
        new_V = V + self.learning_rate * delta_V

        return new_W, new_V

    def _get_class_from_prediction(self, pred) -> np.array:
        return np.where(pred < np.mean(self.classes_),
                        self.classes_[0], self.classes_[1])

    def _mean_square_error(self, pred, y) -> float:
        return np.mean(np.square(y - pred))

    def _missclassification_ratio(self, pred, y) -> float:
        return np.sum(pred != y)/len(y)

    def fit(self, X: np.array, y: np.array) -> None:
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.n_outputs_ = 1 if n_classes == 2 else n_classes

        self.prev_delta_W_ = 0
        self.prev_delta_V_ = 0

        if self.validation_fraction != 0:
            data = np.hstack((X, y.reshape((-1, 1))))

            np.random.shuffle(data)
            index = int(self.validation_fraction * data.shape[0])
            data_train, data_val = data[index:, :], data[:index, :]

            X_train, y_train = data_train[:, :-
                                          1].transpose(), data_train[:, -1]
            X_val, y_val = data_val[:, :-1].transpose(), data_val[:, -1]

            X_val = np.pad(X_val, pad_width=((0, 1), (0, 0)),
                           mode='constant', constant_values=1)
        else:
            X_train = X.transpose()
            y_train = y

        X_train = np.pad(X_train, pad_width=((0, 1), (0, 0)),
                         mode='constant', constant_values=1)
        W = np.random.normal(size=(self.hidden_layer_size, X_train.shape[0]))
        V = np.random.normal(
            size=(self.n_outputs_, self.hidden_layer_size + 1))

        self.loss_curve_ = []
        self.val_loss_curve_ = []
        for epoch in range(self.max_iterations):
            H, O = self._forward_pass(X_train, W, V)
            delta_H, delta_O = self._backward_pass(X_train, y_train, H, O, V)
            W, V = self._weight_update(X_train, W, V, H, delta_H, delta_O)

            _, pred = self._forward_pass(X_train, W, V)
            pred = self._get_class_from_prediction(pred[0])
            self.loss_curve_.append(self._mean_square_error(
                pred, y_train))

            if self.validation_fraction != 0:
                _, pred_val = self._forward_pass(X_val, W, V)
                self.val_loss_curve_.append(self._mean_square_error(
                    self._get_class_from_prediction(pred_val[0]), y_val))

        self.coefs_ = list([W[:, :-1], V[:, :-1]])
        self.intercepts_ = list([W[:, -1], V[:, -1]])

    def predict(self, X: np.array, y: np.array) -> np.array:
        X = np.pad(X.transpose(), pad_width=((0, 1), (0, 0)),
                   mode='constant', constant_values=1)
        W = np.hstack((self.coefs_[0], self.intercepts_[
            0].reshape([-1, 1])))
        V = np.hstack((self.coefs_[1], self.intercepts_[
            1].reshape([-1, 1])))

        _, pred_proba = self._forward_pass(X, W, V)
        pred = self._get_class_from_prediction(pred_proba)
        self.loss_ = self._mean_square_error(pred, y)

        return pred

    def get_decision_boundary(self) -> np.array:
        return None
