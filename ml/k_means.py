import numpy as np


def find_closest_centroids(X, centroids):
    """
    Returns np.array of length m, where each value is index of centroid, to which the i-th example of X is the closest
    :param X: shape m x n
    :param centroids: shape k x n
    :return:
    """
    (m, n) = X.shape
    assert n == centroids.shape[1]
    k = centroids.shape[0]
    closest_centroids = np.empty(m, dtype=np.int32)
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


def k_means(X, centroids, iterations=10, listener_fun=None):
    (m, n) = X.shape
    assert n == centroids.shape[1]
    k = centroids.shape[0]

    previous_centroids = centroids

    for i in range(iterations):
        closest_centroids = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, closest_centroids, k)
        if listener_fun:
            listener_fun(previous_centroids, centroids)
        previous_centroids = centroids

    return centroids


def init_random_centroids(X, k):
    rng = np.random.default_rng()
    X_random_rows = rng.permutation(X)
    return X_random_rows[:k, :]
