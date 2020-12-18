import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import image

import ml.k_means as k_means


def k_means_warm_up():
    data2 = sio.loadmat("ml_course_solutions/machine-learning-ex7/ex7/ex7data2.mat")
    X = data2["X"]
    centroids = np.array([[3, 3], [6, 2], [8, 5]])
    closest_centroids = k_means.find_closest_centroids(X, centroids)
    print(closest_centroids[:3])
    print(k_means.compute_centroids(X, closest_centroids, 3))

    centroids = k_means.init_random_centroids(X, 3)
    (centroids, closest_centroids) = k_means.k_means(X, centroids, listener_fun=plot_k_means)
    plt.plot(X[closest_centroids == 0, 0], X[closest_centroids == 0, 1], "r+")
    plt.plot(X[closest_centroids == 1, 0], X[closest_centroids == 1, 1], "g+")
    plt.plot(X[closest_centroids == 2, 0], X[closest_centroids == 2, 1], "b+")
    plt.show()


def plot_k_means(previous_centroids, centroids):
    k = len(centroids)
    for i in range(k):
        plt.plot([previous_centroids[i, 0], centroids[i, 0]], [previous_centroids[i, 1], centroids[i, 1]], "kx-")


def k_means_picture():
    bird_image = image.imread("ml_course_solutions/machine-learning-ex7/ex7/bird_small.png")
    plt.imshow(bird_image)
    plt.show(block=False)

    im_shape = bird_image.shape

    X = bird_image.reshape([im_shape[0] * im_shape[1], 3])

    (centroids, closest_centroids) = k_means.k_means(X, k_means.init_random_centroids(X, 16))

    X_compressed = np.empty(X.shape)
    for pixel_idx in range(X.shape[0]):
        X_compressed[pixel_idx, :] = centroids[closest_centroids[pixel_idx]]
    plt.imshow(X_compressed.reshape(im_shape))
    plt.show()


if __name__ == "__main__":
    k_means_warm_up()
    k_means_picture()
