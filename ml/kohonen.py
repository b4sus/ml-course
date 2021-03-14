import time

import numpy as np


class KohonenMap:

    def __init__(self, map_length, init_learning_rate=1, learning_rate_constant=3000, init_sigma=1.2,
                 sigma_constant=4000, max_iter=None, observer=None):
        """

        :param map_length:
        :param init_learning_rate:
        :param learning_rate_constant:
        :param init_sigma: 1/4 of map_length seems ok
        :param sigma_constant:
        :param max_iter:
        """
        self.map_length = map_length
        self.rng = np.random.default_rng()
        self.init_learning_rate = init_learning_rate
        self.learning_rate_constant = learning_rate_constant
        self.init_sigma = init_sigma
        self.sigma_constant = sigma_constant
        self.max_iter = max_iter if max_iter is not None else 500 * map_length * map_length
        self.observer = observer

    def fit(self, X, *args, **kwargs):
        X_repr = kwargs["X_repr"]
        nr_features = X.shape[1]
        self.Thetas = self.rng.uniform(size=(self.map_length, self.map_length, nr_features))
        self.iteration = 0
        learning_rate = self.init_learning_rate
        sigma = self.init_sigma

        # self.observer(X=X, map_length=self.map_length, iter=self.iteration, Thetas=self.Thetas,
        #               learning_rate=learning_rate, sigma=sigma, X_repr=X_repr)

        self.coords = np.empty((self.map_length, self.map_length, 2))
        for neuron_x in range(self.map_length):
            for neuron_y in range(self.map_length):
                self.coords[neuron_x, neuron_y] = np.array([neuron_x, neuron_y])

        while self.iteration < self.max_iter:
            x = X[self.rng.integers(0, len(X))]
            bmu_coords = self.find_bmu_using_np(x)
            Deltas, neighbourhoods = self.find_deltas_using_np(x, bmu_coords, learning_rate, sigma, nr_features)

            self.Thetas += Deltas

            self.iteration += 1
            learning_rate = self.learning_rate()
            sigma = self.sigma()

            self.observer(X=X, map_length=self.map_length, iter=self.iteration, Thetas=self.Thetas,
                          learning_rate=learning_rate, sigma=sigma,
                          neighbour_change=self.direct_neighbour(neighbourhoods, bmu_coords), X_repr=X_repr)
        return self

    def transform(self, X, *args, **kwargs):
        return X

    def find_deltas_using_np(self, x, bmu_coords, learning_rate, sigma, nr_features):
        D = np.linalg.norm(np.array(bmu_coords) - self.coords, axis=2)
        N = np.exp(-(D ** 2) / (sigma ** 2))
        Deltas = np.empty((self.map_length, self.map_length, nr_features))
        for neuron_x in range(self.map_length):
            for neuron_y in range(self.map_length):
                Deltas[neuron_x, neuron_y] = learning_rate * N[neuron_x, neuron_y] * (x - self.Thetas[neuron_x, neuron_y])
        return Deltas, N

    def find_deltas_using_loops(self, x, bmu_coords, learning_rate, sigma, nr_features):
        Deltas = np.empty((self.map_length, self.map_length, nr_features))
        Neighbourhoods = np.empty((self.map_length, self.map_length))
        for neuron_x in range(self.map_length):
            for neuron_y in range(self.map_length):
                n = self.neighbourhood(bmu_coords, (neuron_x, neuron_y), sigma)
                Neighbourhoods[neuron_x, neuron_y] = n
                Deltas[neuron_x, neuron_y] = learning_rate * n * (x - self.Thetas[neuron_x, neuron_y])
        return Deltas, Neighbourhoods

    def find_bmu_using_np(self, x):
        return np.unravel_index(np.argmin(np.linalg.norm(self.Thetas - x, axis=2)), (self.map_length, self.map_length))

    def find_bmu_using_loops(self, x):
        bmu_score = np.inf
        bmu_coords = None
        for neuron_x in range(self.map_length):
            for neuron_y in range(self.map_length):
                theta = self.Thetas[neuron_x, neuron_y]
                # score = np.linalg.norm(x - theta)
                # no need to root square - slightly faster
                score = ((x - theta) ** 2).sum()
                if score < bmu_score:
                    bmu_coords = (neuron_x, neuron_y)
                    bmu_score = score
        return bmu_coords

    def direct_neighbour(self, neighbourhoods, bmu_coords):
        bmu_x, bmu_y = bmu_coords
        assert neighbourhoods[bmu_x, bmu_y] == 1
        if bmu_x < self.map_length - 1:
            neighbour_x = bmu_x + 1
        elif bmu_x > 0:
            neighbour_x = bmu_x - 1
        return neighbourhoods[neighbour_x, bmu_y]

    def learning_rate(self):
        return self.init_learning_rate * np.exp(-self.iteration / self.learning_rate_constant)

    def neighbourhood(self, bmu_coords, current_neuron, sigma):
        d = np.linalg.norm(np.array(bmu_coords) - np.array(current_neuron))
        return np.exp(-(d ** 2) / (sigma ** 2))

    def sigma(self):
        return self.init_sigma * np.exp(-self.iteration / self.sigma_constant)
