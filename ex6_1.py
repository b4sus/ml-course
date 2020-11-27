import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

import ml.kernel as kernel

data1 = sio.loadmat("ml_course_solutions/machine-learning-ex6/ex6/ex6data1.mat")

print(data1)

X1 = data1["X"]
y1 = data1["y"]


def plot(figure, X, y):
    success_indices = (y == 1).flatten()
    fail_indices = (y == 0).flatten()

    plt.figure(figure)
    plt.plot(X[success_indices, 0], X[success_indices, 1], "bx")
    plt.plot(X[fail_indices, 0], X[fail_indices, 1], "yo")

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

gaussianKernelExpectedValue = 0.324652
gaussianKernelTestValue = kernel.gaussian_kernel(np.array([[1], [2], [1]]), np.array([[0], [4], [-1]]), 2)

print(f"expected gaussian distance {gaussianKernelExpectedValue}, actual value: {gaussianKernelTestValue}")

data2 = sio.loadmat("ml_course_solutions/machine-learning-ex6/ex6/ex6data2.mat")

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