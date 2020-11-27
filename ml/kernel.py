import numpy as np


class GaussianKernelForSklearn:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, *args, **kwargs):
        X = args[0]
        (m, n) = X.shape

        X_original = args[1]

        (m_original, n_original) = X_original.shape

        assert n == n_original

        X_gaussian = np.empty((m, m_original))

        for row_idx in range(m):
            for landmark_idx in range(m_original):
                X_gaussian[row_idx, landmark_idx] = gaussian_kernel(X[row_idx, :], X_original[landmark_idx, :],
                                                                    self.sigma)

        return X_gaussian


def gaussian_kernel(x, landmark, sigma):
    assert x.shape == landmark.shape

    return np.exp(- (np.linalg.norm(x - landmark) ** 2) / (2 * (sigma ** 2)))
