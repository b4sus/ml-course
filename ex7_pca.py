import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import image

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

    centroids = k_means.k_means(X, k_means.init_random_centroids(X, 16))

    closest_centroids = k_means.find_closest_centroids(X, centroids)


if __name__ == "__main__":
    # pca_warmup()
    # pca_faces()
    pca_on_bird()
