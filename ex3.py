import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import ml.pipeline as pipeline
import ml.predict as predict
import ml.ml as ml

images_mat = sio.loadmat("data/ex3data1.mat")

X = images_mat["X"]
y = images_mat["y"]
y[y == 10] = 0

indices = {}
thetas = {}
y_flat = y.flatten()

for i in range(10):
    y_i = np.zeros(5000)
    y_i[y_flat == i] = 1
    pipeline_i = pipeline.Pipeline()
    pipeline_i.bias()
    (theta_i, X_i) = pipeline_i.execute_train(X, y_i.reshape((5000, 1)), regularization=0.5)
    thetas[i] = theta_i

hypotheses = np.zeros((5000, 10))

for (digit, theta) in thetas.items():
    prediction = predict.predict(X, theta, ml.logistic_regression_hypothesis)
    hypotheses[:, digit] = ml.logistic_regression_hypothesis(np.hstack((np.ones((5000, 1)), X)), theta)[:, 0]

predictions = np.argmax(hypotheses, axis=1).reshape((5000, 1))

print(np.mean(predictions == y))



# plt.imshow(X[2500].reshape((20, 20)).T, cmap='gray')
# plt.show()