import scipy.io as sio
import matplotlib.pyplot as plt

import ml.feature as feature
import ml.pca as pca


def pca_warmup():
    data = sio.loadmat("ml_course_solutions/machine-learning-ex7/ex7/ex7data1.mat")
    X = data["X"]
    # plt.plot(X[:, 0], X[:, 1], '.')
    # plt.show(block=False)

    normalizer = feature.FeatureNormalizer(X)
    X_norm = normalizer.normalized_x_m
    U0   = pca.pca(X_norm)

    plt.plot(X_norm[:, 0], X_norm[:, 1], '.')
    plt.show(block=False)

    plt.plot([-1, U0[0]], [-1, U0[1]], "x-")
    plt.show()


if __name__ == "__main__":
    pca_warmup()
