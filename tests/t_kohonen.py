

import numpy as np

import ml.kohonen as kh


def kohonen_square_5():
    km = kh.KohonenMap(5, 2)
    rng = np.random.default_rng()
    X = rng.uniform(size=(1000, 2))

    km.fit(X)


def kohonen_square_10():
    km = kh.KohonenMap(10, 2, init_learning_rate=1, learning_rate_constant=3000, init_sigma=2.5, sigma_constant=3000)
    rng = np.random.default_rng()
    X = rng.uniform(size=(1000, 2))

    km.fit(X)


def kohonen_square_20():
    km = kh.KohonenMap(20, 2, init_learning_rate=1, learning_rate_constant=12000, init_sigma=5, sigma_constant=12000)
    rng = np.random.default_rng()
    X = rng.uniform(size=(10000, 2))

    km.fit(X)


if __name__ == "__main__":
    kohonen_square_20()
