import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

from ml.anomaly_detection import gaussian, estimate_gaussian, select_threshold
import ml.pca as pca
import ml.feature as feature


def visualize(X, figure=0, *, block=False):
    plt.figure(figure)
    plt.plot(X[:, 0], X[:, 1], "x")
    plt.show(block=block)


def visualize_fit(X, mu, sigma2, *, block=False):
    ls = np.linspace(0, 30, 100)
    (mesh_x, mesh_y) = np.meshgrid(ls, ls)
    Z = gaussian(np.c_[mesh_x.flatten(), mesh_y.flatten()], mu, sigma2)
    plt.contour(mesh_x, mesh_y, Z.reshape((100, 100)), [10 ** i for i in range(-20, 0, 3)])
    plt.show(block=block)


def server_anomaly_detection():
    data = sio.loadmat("ml_course_material/machine-learning-ex8/ex8/ex8data1.mat")
    X = data["X"]
    X_cv = data["Xval"]
    y_cv = data["yval"]

    visualize(X, block=False)
    (mu, sigma2) = estimate_gaussian(X)
    visualize_fit(X, mu, sigma2, block=False)
    p_cv = gaussian(X_cv, mu, sigma2)
    (epsilon, f1) = select_threshold(p_cv, y_cv)
    print(f"best epsilon is {epsilon} with F1 score {f1}")
    p = gaussian(X, mu, sigma2)
    (anomalies_indices_x, anomalies_indices_y) = np.nonzero(p < epsilon)
    plt.plot(X[anomalies_indices_x, 0], X[anomalies_indices_x, 1], '+')
    plt.show()


def anomaly_detection_high_dimensional_dataset():
    data = sio.loadmat("ml_course_material/machine-learning-ex8/ex8/ex8data2.mat")
    X = data["X"]
    X_cv = data["Xval"]
    y_cv = data["yval"]

    (mu, sigma2) = estimate_gaussian(X)
    p_cv = gaussian(X_cv, mu, sigma2)
    (epsilon, f1) = select_threshold(p_cv, y_cv)
    print(f"best epsilon is {epsilon} with F1 score {f1}")
    p = gaussian(X, mu, sigma2)
    (anomalies_indices_x, anomalies_indices_y) = np.nonzero(p < epsilon)
    print(f"{len(anomalies_indices_x)} anomalies")

    # pca for fun
    normnalizer = feature.FeatureNormalizer(X)
    X_norm = normnalizer.normalized_x_m
    (U, S) = pca.pca(X_norm)
    Z = pca.project(X_norm, U, 2)
    visualize(Z)
    plt.plot(Z[anomalies_indices_x, 0], Z[anomalies_indices_x, 1], '+')
    plt.show()


if __name__ == "__main__":
    # anomaly_detection_high_dimensional_dataset()
    server_anomaly_detection()