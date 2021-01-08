import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn import svm

import ml.kernel as kernel


def plot(figure, X, y):
    success_indices = (y == 1).flatten()
    fail_indices = (y == 0).flatten()

    plt.figure(figure)
    plt.plot(X[success_indices, 0], X[success_indices, 1], "bx")
    plt.plot(X[fail_indices, 0], X[fail_indices, 1], "yo")


def dataset1():
    data1 = sio.loadmat("ml_course_material/machine-learning-ex6/ex6/ex6data1.mat")

    print(data1)

    X1 = data1["X"]
    y1 = data1["y"]

    plot(0, X1, y1)

    svc = svm.SVC(C=1, kernel="linear")

    svc.fit(X1, y1.flatten())

    print(svc)

    x1_axis = np.linspace(0, 4, 100)
    x2_axis = np.linspace(1.5, 5, 100)

    mesh_x1, mesh_x2 = np.meshgrid(x1_axis, x2_axis)

    X_total_space = np.c_[mesh_x1.reshape(-1), mesh_x2.reshape(-1)]

    Z = svc.predict(X_total_space)

    plt.contourf(mesh_x1, mesh_x2, Z.reshape((100, 100)), alpha=0.2)
    plt.show()


def gaussianKernelTest():
    gaussianKernelExpectedValue = 0.324652
    gaussianKernelTestValue = kernel.gaussian_kernel(np.array([[1], [2], [1]]), np.array([[0], [4], [-1]]), 2)

    print(f"expected gaussian distance {gaussianKernelExpectedValue}, actual value: {gaussianKernelTestValue}")


def dataset2():
    data2 = sio.loadmat("ml_course_material/machine-learning-ex6/ex6/ex6data2.mat")

    X2 = data2["X"]
    y2 = data2["y"]

    plot(1, X2, y2)

    svc = svm.SVC(C=1, kernel=kernel.GaussianKernelForSklearn(0.1))
    # svc = svm.SVC(C=1, gamma=100)
    svc.fit(X2, y2.flatten())

    x1_axis = np.linspace(0, 1, 100)
    x2_axis = np.linspace(0.4, 1, 100)

    mesh_x1, mesh_x2 = np.meshgrid(x1_axis, x2_axis)

    X_total_space = np.c_[mesh_x1.reshape(-1), mesh_x2.reshape(-1)]

    Z = svc.predict(X_total_space)

    plt.contourf(mesh_x1, mesh_x2, Z.reshape((100, 100)), alpha=0.2)
    plt.show()


def dataset3():
    data = sio.loadmat("ml_course_material/machine-learning-ex6/ex6/ex6data3.mat")
    X = data["X"]
    y = data["y"]
    X_cv = data["Xval"]
    y_cv = data["yval"]
    plot(2, X, y)
    plt.show()

    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    errors = {}

    for C in Cs:
        for sigma in sigmas:
            svc = svm.SVC(C=C, kernel=kernel.GaussianKernelForSklearn(sigma))
            svc.fit(X, y.flatten())
            y_cv_predicted = svc.predict(X_cv).reshape((-1, 1))
            errors[(C, sigma)] = np.mean(y_cv != y_cv_predicted)

    print(errors)

    sorted_errors = sorted(errors.items(), key=lambda keyValue: keyValue[1])
    print(f"sorted errors ((C, sigma), accuracy) {sorted_errors}")

    winner = sorted_errors[0]

    print(f"winner {winner}")

    svc = svm.SVC(C=winner[0][0], kernel=kernel.GaussianKernelForSklearn(winner[0][1]))
    svc.fit(X, y.flatten())

    plot(2, X, y)
    x_axis = np.linspace(-0.6, 0.4, 100)
    y_axis = np.linspace(-0.7, 0.7, 100)

    mesh_x, mesh_y = np.meshgrid(x_axis, y_axis)

    X_total_space = np.c_[mesh_x.reshape(-1), mesh_y.reshape(-1)]

    Z = svc.predict(X_total_space)

    plt.contourf(mesh_x, mesh_y, Z.reshape((100, 100)), alpha=0.2)
    plt.show()


if __name__ == "__main__":
    dataset3()
