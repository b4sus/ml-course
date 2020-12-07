import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def find_closest_centroids(X, centroids):
    (m, n) = X.shape
    assert n == centroids.shape[1]
    k = centroids.shape[0]
    closest_centroids = np.empty(m)
    for i in range(m):
        closest_distance = None
        for centroid_idx in range(k):
            distance = np.linalg.norm(X[i, :] - centroids[centroid_idx, :]) ** 2
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
                closest_centroids[i] = centroid_idx
    return closest_centroids


def compute_centroids(X, closest_centroid_indices, k):
    (m, n) = X.shape
    assert m == len(closest_centroid_indices)
    new_centroids = np.empty((k, n))
    for centroid_idx in range(k):
        indices = closest_centroid_indices == centroid_idx
        C = X[indices, :]
        new_centroids[centroid_idx, :] = C.sum(0) / len(C)
    return new_centroids


def k_means(X, centroids):
    (m, n) = X.shape
    assert n == centroids.shape[1]
    k = centroids.shape[0]

    previous_centroids = centroids

    for i in range(10):
        closest_centroids = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, closest_centroids, k)
        plot_k_means(X, centroids, previous_centroids)
        previous_centroids = centroids

    closest_centroids = find_closest_centroids(X, centroids)
    plt.plot(X[closest_centroids == 0, 0], X[closest_centroids == 0, 1], "rx")
    plt.plot(X[closest_centroids == 1, 0], X[closest_centroids == 1, 1], "gx")
    plt.plot(X[closest_centroids == 2, 0], X[closest_centroids == 2, 1], "bx")
    plt.show()
    return centroids


def plot_k_means(X, centroids, previous_centroids):
    k = len(centroids)
    for i in range(k):
        plt.plot([previous_centroids[i, 0], centroids[i, 0]], [previous_centroids[i, 1], centroids[i, 1]], "ko-")


if __name__ == "__main__":
    data2 = sio.loadmat("ml_course_solutions/machine-learning-ex7/ex7/ex7data2.mat")
    X = data2["X"]
    centroids = np.array([[3, 3], [6, 2], [8, 5]])
    closest_centroids = find_closest_centroids(X, centroids)
    print(closest_centroids[:3])
    print(compute_centroids(X, closest_centroids, 3))
    k_means(X, centroids)
