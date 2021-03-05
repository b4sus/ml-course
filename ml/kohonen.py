import time

import matplotlib.pyplot as plt
import numpy as np


class KohonenMap:

    def __init__(self, map_length, m, init_learning_rate=1, learning_rate_constant=3000, init_sigma=1.2,
                 sigma_constant=4000, max_iter=None):
        """

        :param map_length:
        :param m:
        :param init_learning_rate:
        :param learning_rate_constant:
        :param init_sigma: 1/4 of map_length seems ok
        :param sigma_constant:
        :param max_iter:
        """
        self.map_length = map_length
        self.m = m
        self.rng = np.random.default_rng()
        self.Thetas = self.rng.uniform(size=(self.map_length, self.map_length, self.m))
        self.init_learning_rate = init_learning_rate
        self.learning_rate_constant = learning_rate_constant
        self.init_sigma = init_sigma
        self.sigma_constant = sigma_constant
        self.max_iter = max_iter if max_iter is not None else 500 * map_length * map_length
        self.figure = 0

    def fit(self, X):
        self.iteration = 0
        learning_rate = self.init_learning_rate
        sigma = self.init_sigma
        self.draw(X, learning_rate, sigma)
        while self.iteration < self.max_iter:
            X = self.rng.permutation(X, axis=0)
            for i, x in enumerate(X):
                # print(f"Starting it {self.iteration} with learning rate {learning_rate} and sigma {sigma}")
                best_score = np.inf
                best_neuron = None
                for neuron_x in range(self.map_length):
                    for neuron_y in range(self.map_length):
                        theta = self.Thetas[neuron_x, neuron_y]
                        score = np.linalg.norm(x - theta)
                        if score < best_score:
                            best_neuron = (neuron_x, neuron_y)
                            best_score = score
                neighbourhoods = np.empty((self.map_length, self.map_length))
                for neuron_x in range(self.map_length):
                    for neuron_y in range(self.map_length):
                        n = self.neighbourhood(best_neuron, (neuron_x, neuron_y), sigma)
                        neighbourhoods[neuron_x, neuron_y] = n
                        self.Thetas[neuron_x, neuron_y] += learning_rate * n * (x - self.Thetas[neuron_x, neuron_y])

                self.iteration += 1
                learning_rate = self.learning_rate()
                sigma = self.sigma()
                if self.iteration % 10 == 0:
                    self.draw(X, learning_rate, sigma, neighbour_change=self.direct_neighbour(neighbourhoods, best_neuron))

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
        # return 1 / self.iteration

    def neighbourhood(self, winner_neuron, current_neuron, sigma):
        d = np.linalg.norm(np.array(winner_neuron) - np.array(current_neuron))
        return np.exp(-(d ** 2) / (sigma ** 2))

    def sigma(self):
        return self.init_sigma * np.exp(-self.iteration / self.sigma_constant)

    def draw(self, X, learning_rate, sigma, selected_x=None, winner_neuron=None, neighbour_change=-1):
        plt.figure(0)
        plt.figure(0).clear()
        plt.title(
            f"Iteration {self.iteration} - learning rate {learning_rate:.3f}, sigma {sigma:.3f}, neighbour_change {neighbour_change:.3f}")
        plt.scatter(X[:, 0], X[:, 1], marker="x")
        if selected_x is not None:
            plt.scatter(selected_x[0], selected_x[1], marker="x", c="r")
        plt.scatter(self.Thetas[:, :, 0], self.Thetas[:, :, 1])
        if winner_neuron is not None:
            plt.scatter(winner_neuron[0], winner_neuron[1], c="g")
        # plt.annotate(self.Thetas[:, :, 0], self.Thetas[:, :, 1], self.Thetas[:, :, 0] + self.Thetas[:, :, 1])
        for neuron_x in range(self.map_length):
            for neuron_y in range(self.map_length):
                x = self.Thetas[neuron_x, neuron_y, 0]
                y = self.Thetas[neuron_x, neuron_y, 1]
                plt.annotate(f"{neuron_x} {neuron_y}", (x, y))
                if neuron_x > 0:
                    plt.plot([x, self.Thetas[neuron_x - 1, neuron_y, 0]], [y, self.Thetas[neuron_x - 1, neuron_y, 1]],
                             "k")
                if neuron_x < self.map_length - 1:
                    plt.plot([x, self.Thetas[neuron_x + 1, neuron_y, 0]], [y, self.Thetas[neuron_x + 1, neuron_y, 1]],
                             "k")
                if neuron_y > 0:
                    plt.plot([x, self.Thetas[neuron_x, neuron_y - 1, 0]], [y, self.Thetas[neuron_x, neuron_y - 1, 1]],
                             "k")
                if neuron_y < self.map_length - 1:
                    plt.plot([x, self.Thetas[neuron_x, neuron_y + 1, 0]], [y, self.Thetas[neuron_x, neuron_y + 1, 1]],
                             "k")
        plt.show(block=False)
        plt.figure(0).canvas.flush_events()
        time.sleep(0.01)
