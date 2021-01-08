import numpy as np
import scipy.io as sio

import ml.pipeline as pipeline
import ml.predict as predict
import ml.logistic_regression as lore

images_mat = sio.loadmat("ml_course_material/machine-learning-ex3/ex3/ex3data1.mat")

X = images_mat["X"]
y = images_mat["y"]
y[y == 10] = 0

y_flat = y.flatten()

pipeline = pipeline.Pipeline()
pipeline.bias()

thetas = pipeline.one_vs_all(X, y)

hypotheses = np.zeros((5000, 10))

for (digit, theta) in enumerate(thetas):
    prediction = predict.predict(X, theta, lore.logistic_regression_hypothesis)
    hypotheses[:, digit] = lore.logistic_regression_hypothesis(np.hstack((np.ones((5000, 1)), X)), theta)[:, 0]

predictions = np.argmax(hypotheses, axis=1).reshape((5000, 1))

print(np.mean(predictions == y))



# plt.imshow(X[2500].reshape((20, 20)).T, cmap='gray')
# plt.show()