import numpy as np


def predict(x_m, theta, hypothesis, normalizer=None):
    (m, n) = x_m.shape

    assert theta.shape == (n + 1, 1), "Expecting x_m not to include the 1s as the first column"

    if normalizer:
        x_m = normalizer.normalize(x_m)

    x_m_normalized = np.hstack((np.ones((m, 1)), x_m))
    predicted_probabilities = hypothesis(x_m_normalized, theta)
    predicter = lambda x: 1 if x > 0.5 else 0
    return np.vectorize(predicter)(predicted_probabilities)
