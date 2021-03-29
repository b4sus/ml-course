# mlplay
My rework of th machine learning course from andrew ng into python

* ml/linear_regression.py - cost and gradient function for linear regression
* ml/gradient_descent.py - gradient descent function using provided cost and gradient function
* ml/logistic_regression.py - cost and gradient function for logistic regression
* ml/neural_network.py - forward feed of nn and backpropagation
* ml/learning_curves.py - functions helpful to visualize learning of linear regression with different learning rates/training set size/polynomial degree
* ml/kernel.py - gaussian kernel implementation usable in sklearn.svm.SVC
* ml/k_means.py - implementation of k-means clustering
* ml/pca.py - pca implementation using svd from numpy.linalg
* ml/anomaly_detection.py - computing gaussian distribution and threshold selection for detecting an anomaly
* ml/collaborative_filtering.py - cost and gradient functions for collaborative filtering

In the root folder there are runner scripts named like ex1.py for running the specific exercises. They are very rough equivalents to the course originals ex1.m files. They will not work as I didn't commit the datasets (they are not mine to spread). 

# TODOs
experimenting with housing prices and learning curves - trying to make polynomial features completely,
not just numerical