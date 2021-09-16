import numpy as np
from typing import Tuple


class TwoLayerPerceptron:
    learning_rate: float
    momentum: float
    max_iterations: int
    tolerance: float
    hidden_layer_size: int
    validation_fraction: float
    is_classification_task: bool
    mode: str

    classes_: np.array
    n_outputs_: int
    W_ = np.array
    V_ = np.array
    prev_delta_W_: float
    prev_delta_V_: float
    error_per_epoch: dict
    error_per_epoch_val: dict
    loss_: dict

    def __init__(
        self,
        mode: str = 'batch',
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        max_iterations: int = 200,
        tolerance: float = 0,
        hidden_layer_size: int = 1,
        validation_fraction: float = 0,
        is_classification_task: bool = True
    ):
        self.mode = mode
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.hidden_layer_size = hidden_layer_size
        self.validation_fraction = validation_fraction
        self.is_classification_task = is_classification_task
        self._reset()

    def _reset(self) -> None:
        if self.is_classification_task:
            self.error_per_epoch = {
                'missclassification_error': [],
                'mse': []
            }

            self.error_per_epoch_val = {
                'missclassification_error': [],
                'mse': []
            }

            self.loss_ = {
                'missclassification_error': None,
                'mse': None
            }
        else:
            self.error_per_epoch = {
                'mse': []
            }

            self.error_per_epoch_val = {
                'mse': []
            }

            self.loss_ = {
                'mse': None
            }

        self.prev_delta_W_ = 0
        self.prev_delta_V_ = 0

    def _pad(self, X: np.array) -> np.array:
        return np.pad(X, pad_width=((0, 1), (0, 0)),
                      mode='constant', constant_values=1)

    def _transfer_function(self, x: np.array) -> np.array:
        return 2/(1 + np.exp(-x)) - 1

    def _transfer_function_derivative(self, calculated_transfer_function: np.array) -> np.array:
        return np.multiply((1 + calculated_transfer_function), (1 - calculated_transfer_function))/2

    def _forward_pass(self, X: np.array, W: np.array, V: np.array) -> Tuple[np.array]:
        H = self._transfer_function(W @ X)
        H = self._pad(H)
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

    def _evaluate_performance(self, X, y, W, V, error) -> None:
        _, pred = self._forward_pass(X, W, V)

        error['mse'].append(self._mean_square_error(
            pred, y))

        if self.is_classification_task:
            pred_class = self._get_class_from_prediction(pred[0])
            error['missclassification_error'].append(self._misclassification_ratio(
                pred_class, y))

    def _get_class_from_prediction(self, pred) -> np.array:
        return np.where(pred < np.mean(self.classes_),
                        self.classes_[0], self.classes_[1])

    def _mean_square_error(self, pred, y) -> float:
        return np.mean(np.square(y - pred))

    def _misclassification_ratio(self, pred, y) -> float:
        return np.sum(pred != y)/len(y)

    def fit(self, X: np.array, y: np.array, X_val: np.array = None, y_val: np.array = None) -> None:
        if self.is_classification_task:
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            self.n_outputs_ = 1 if n_classes == 2 else n_classes
        else:
            self.n_outputs_ = 1

        self._reset()

        X_train = X
        y_train = y
        if self.validation_fraction != 0:
            data = np.hstack((X, y.reshape((-1, 1))))

            np.random.shuffle(data)
            index = int(self.validation_fraction * data.shape[0])
            data_train, data_val = data[index:, :], data[:index, :]

            X_train, y_train = data_train[:, :-
                                          1], data_train[:, -1]
            X_val, y_val = data_val[:, :-1], data_val[:, -1]

        if X_val is not None:
            X_val = X_val.transpose()
            X_val = self._pad(X_val)
        X_train = X_train.transpose()
        X_train = self._pad(X_train)

        W = np.random.normal(size=(self.hidden_layer_size, X_train.shape[0]))
        V = np.random.normal(
            size=(self.n_outputs_, self.hidden_layer_size + 1))

        # if self.validation_fraction != 0 or (X_val is not None and y_val is not None):
        #     print('X_val ', X_val.shape)
        #     print('y_val ', y_val.shape)

        # print('X_train ', X_train.shape)
        # print('y_train ', y_train.shape)
        # print('W ', W.shape)
        # print('V ', V.shape)
        for epoch in range(self.max_iterations):
            if self.mode == 'batch':
                H, O = self._forward_pass(X_train, W, V)
                delta_H, delta_O = self._backward_pass(
                    X_train, y_train, H, O, V)
                W, V = self._weight_update(X_train, W, V, H, delta_H, delta_O)
            elif self.mode == 'online':
                for index in range(X_train.shape[1]):
                    X_curr = X_train[:, index].reshape((-1, 1))
                    y_curr = y_train[index].reshape((-1, 1))

                    H, O = self._forward_pass(X_curr, W, V)
                    delta_H, delta_O = self._backward_pass(
                        X_curr, y_curr, H, O, V)
                    W, V = self._weight_update(
                        X_curr, W, V, H, delta_H, delta_O)

            self._evaluate_performance(
                X_train, y_train, W, V, self.error_per_epoch)

            if self.validation_fraction != 0 or (X_val is not None and y_val is not None):
                self._evaluate_performance(
                    X_val, y_val, W, V, self.error_per_epoch_val)

        self.W_ = W.copy()
        self.V_ = V.copy()

    def predict(self, X: np.array, y: np.array = None) -> np.array:
        X = self._pad(X.transpose())
        W = np.hstack((self.coefs_[0], self.intercepts_[
            0].reshape([-1, 1])))
        V = np.hstack((self.coefs_[1], self.intercepts_[
            1].reshape([-1, 1])))

        _, pred_proba = self._forward_pass(X, W, V)
        pred = self._get_class_from_prediction(pred_proba)

        if not y is None:
            if self.is_classification_task:
                self.loss_ = {
                    'missclassification_error': self._misclassification_ratio(pred, y),
                    'mse': self._mean_square_error(pred, y)
                }
            else:
                self.loss_ = {
                    'mse': self._mean_square_error(pred, y)
                }

        return pred

    def get_decision_boundary(self) -> np.array:
        return None

    @property
    def intercepts_(self):
        return list([self.W_[:, -1], self.V_[:, -1]])

    @property
    def coefs_(self):
        return list([self.W_[:, :-1], self.V_[:, :-1]])
