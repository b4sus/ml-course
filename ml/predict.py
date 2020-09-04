import numpy as np


def predict(x_m, theta, hypothesis, normalizer=None):
    (m, n) = np.shape(x_m)
    assert np.shape(theta) == (n + 1, 1)
    if normalizer:
        x_m = normalizer.normalize(x_m)
    x_m_normalized = np.hstack((np.ones((m, 1)), x_m))
    predicted_probabilities = hypothesis(x_m_normalized, theta)
    predicter = lambda x: 1 if x > 0.5 else 0
    return np.vectorize(predicter)(predicted_probabilities)
