import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np

import ml.feature as feature
import ml.pca as pca
import ml.k_means as k_means


def pca_warmup():
    data = sio.loadmat("ml_course_solutions/machine-learning-ex7/ex7/ex7data1.mat")
    X = data["X"]
    # plt.plot(X[:, 0], X[:, 1], '.')
    # plt.show(block=False)

    normalizer = feature.FeatureNormalizer(X)
    X_norm = normalizer.normalized_x_m
    (U, S) = pca.pca(X_norm)

    print(U[:, 0])

    plt.plot(X_norm[:, 0], X_norm[:, 1], '.')
    plt.show(block=False)

    plt.plot([-3, U[0, 0]], [-3, U[0, 1]], "x-")
    plt.show(block=False)

    Z = pca.project(X_norm, U, 1)
    print(Z)
    X_approx = pca.reconstruct(Z, U, 1)
    print(X_approx)
    plt.plot(X_approx[:, 0], X_approx[:, 1], "y.")
    plt.show()


def pca_faces():
    data = sio.loadmat("ml_course_solutions/machine-learning-ex7/ex7/ex7faces.mat")
    X = data["X"]

    plt.figure(0)
    plot_images(X)
    plt.show(block=False)  #somehow when plotting, svd will not converge

    normalizer = feature.FeatureNormalizer(X)
    X_norm = normalizer.normalized_x_m

    while True:
        try:
            (U, S) = pca.pca(X_norm)
            break
        except:
            pass


    plt.figure(1)
    plot_images(U.T)
    plt.show(block=False)

    k = 100

    Z = pca.project(X, U, k)

    X_approx = pca.reconstruct(Z, U, k)

    plt.figure(2)
    plot_images(X_approx)
    plt.show()


def plot_images(X):
    for i in range(100):
        subplot = plt.subplot(10, 10, i + 1)
        subplot.set(xticklabels=[])
        subplot.set(yticklabels=[])
        subplot.set(xlabel=None)
        subplot.set(ylabel=None)
        subplot.tick_params(bottom=False, left=False)
        plt.imshow(X[i, :].reshape((32, 32)).T, cmap="gray")


def pca_on_bird():
    bird_image = image.imread("ml_course_solutions/machine-learning-ex7/ex7/bird_small.png")
    plt.imshow(bird_image)
    plt.show(block=False)

    im_shape = bird_image.shape

    X = bird_image.reshape([im_shape[0] * im_shape[1], 3])

    k = 16

    (centroids, closest_centroids) = k_means.k_means(X, k_means.init_random_centroids(X, k))

    nr_samples = 1000
    rng = np.random.default_rng()
    random_indices = rng.integers(0, X.shape[0], nr_samples)

    colors = []
    X_rand = np.empty((nr_samples, X.shape[1]))
    for i in range(nr_samples):
        colors.append(closest_centroids[random_indices[i]])
        X_rand[i, :] = X[random_indices[i], :]

    plt.hsv()
    fig = plt.figure(3)
    ax = fig.gca(projection='3d')
    ax.scatter(X_rand[:, 0], X_rand[:, 1], X_rand[:, 2], c=colors)
    plt.show(block=False)

    X_norm = feature.FeatureNormalizer(X).normalized_x_m
    while True:
        try:
            (U, S) = pca.pca(X_norm)
            break
        except:
            pass

    Z = pca.project(X, U, 2)
    Z_rand = np.empty((nr_samples, Z.shape[1]))
    for i in range(nr_samples):
        Z_rand[i, :] = Z[random_indices[i], :]

    plt.figure(4)
    plt.scatter(Z_rand[:, 0], Z_rand[:, 1], c=colors)
    plt.show()

if __name__ == "__main__":
    # pca_warmup()
    # pca_faces()
    pca_on_bird()
