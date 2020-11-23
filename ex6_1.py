import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

data1 = sio.loadmat("ml_course_solutions/machine-learning-ex6/ex6/ex6data1.mat")

print(data1)

X1 = data1["X"]
y1 = data1["y"]

success_indices = (y1 == 1).flatten()
fail_indices = (y1 == 0).flatten()

plt.plot(X1[success_indices, 0], X1[success_indices, 1], "yo")
plt.plot(X1[fail_indices, 0], X1[fail_indices, 1], "bx")

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