import numpy as np


class KohonenMap:
    def __init__(self, map_length, m):
        self.map_length = map_length
        self.m = m
        self.rng = np.random.default_rng()
        self.Thetas = self.rng.uniform(-0.5, 0.5, (self.map_length, self.map_length, self.m))
        self.init_learning_rate = 1
        self.init_sigma = 1
        self.constant = 10

    def fit(self, X):
        self.iteration = 0
        X = self.rng.permutation(X, axis=0)
        for x in X:
            best_score = np.inf
            best_neuron = None
            for neuron_x in range(self.map_length):
                for neuron_y in range(self.map_length):
                    theta = self.Thetas[neuron_x, neuron_y]
                    score = np.linalg.norm(x - theta)
                    if score < best_score:
                        best_neuron = (neuron_x, neuron_y)
                        best_score = score
            for neuron_x in range(self.map_length):
                for neuron_y in range(self.map_length):
                    self.Thetas[neuron_x, neuron_y] += self.learning_rate()\
                                                       * self.neighbourhood(best_neuron, (neuron_x, neuron_y))\
                                                       * (x - self.Thetas[neuron_x, neuron_y])
            self.iteration += 1

    def learning_rate(self):
        return self.init_learning_rate * np.exp(-self.iteration / self.constant)

    def neighbourhood(self, winner_neuron, current_neuron):
        d = np.linalg.norm(np.array(winner_neuron) - np.array(current_neuron))
        return np.exp(-(d ** 2) / self.sigma() ** 2)

    def sigma(self):
        return self.init_sigma * np.exp(-self.iteration / self.constant)