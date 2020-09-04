import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data/ex2data2.txt", delimiter=",")

print(data)

y = data[:, -1].reshape(len(data), 1)

success_indices = np.nonzero(y.flatten() == 1)
fail_indices = np.nonzero(y.flatten() == 0)

x = data[:, :2]

plt.figure(0)
plt.plot(x[success_indices, 0], x[success_indices, 1], "kx")
plt.plot(x[fail_indices, 0], x[fail_indices, 1], "yo")
plt.show()