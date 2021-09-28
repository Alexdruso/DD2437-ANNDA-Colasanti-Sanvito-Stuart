import numpy as np
from sklearn.base import BaseEstimator
from scipy.spatial import distance
from typing import Tuple


class SOM(BaseEstimator):

    learning_rate: float
    max_iterations: int
    n_nodes: int
    grid_shape: Tuple
    cyclic: bool
    neighborhood_size: int
    grid: np.array

    _W: np.array

    def __init__(
            self,
            learning_rate: float = 0.2,
            n_nodes: int = 100,
            neighborhood_size: int = 50,
            grid_shape: Tuple = None,
            cyclic: bool = False,
            max_iterations: int = 20
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.n_nodes = n_nodes
        self.cyclic = cyclic
        self.neighborhood_size = neighborhood_size
        self.grid_shape = grid_shape

        if grid_shape is None:
            self.grid = np.arange(n_nodes)
        else:
            self.grid = np.indices(grid_shape).transpose(
                1, 2, 0).reshape(-1, 2)

    def _euclidean_distance(self, W: np.array, x: np.array) -> np.array:
        distances = []
        for w in W:
            distances.append(np.sum((x - w) ** 2, axis=0))

        return distances

    def _get_winner(self, distances: np.array) -> int:
        return np.argmin(distances)

    def _get_neighbors(self, winner: int, neighborhood_size: int) -> np.array:
        if self.grid_shape is not None:
            winner = np.unravel_index(winner, self.grid_shape)
            dist = distance.cdist([winner], self.grid, 'cityblock')
            neighbors = np.where(dist <= neighborhood_size)[1]
        else:
            if self.cyclic:
                neighbors = self.grid.take(np.r_[
                    winner - neighborhood_size: winner + neighborhood_size + 1
                ], mode="wrap")
            else:
                neighbors = self.grid.take(np.r_[
                    max(winner - neighborhood_size, 0):min(winner +
                                                           neighborhood_size + 1, self.n_nodes)
                ])
        return neighbors

    def _update_weights(self, x: np.array, W: np.array, indices: np.array) -> None:
        for index in indices:
            W[index, :] = W[index, :] + self.learning_rate * (x - W[index, :])

    def fit(self, X: np.array) -> None:
        W = np.random.uniform(
            size=(self.n_nodes, X.shape[1]))

        neighborhood_size = self.neighborhood_size
        neighborhood_decay = np.ceil(np.linspace(
            neighborhood_size, 0, self.max_iterations)).astype(int)

        for epoch in range(self.max_iterations):
            for x in X:
                distances = self._euclidean_distance(W, x)
                winner = self._get_winner(distances)
                neighbors = self._get_neighbors(winner, neighborhood_size)
                self._update_weights(x, W, neighbors)
            neighborhood_size = neighborhood_decay[epoch]

        self._W = W

    def transform(self, X: np.array, names: np.array = None) -> np.array:
        winners = []
        for x in X:
            distances = self._euclidean_distance(self._W, x)
            winner = self._get_winner(distances)
            winners.append(winner)

        if names is not None:
            sorted_list = [name for _, name in sorted(
                zip(winners, names), key=lambda pair: pair[0])]

            return sorted_list
        else:
            return winners
