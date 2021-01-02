import numpy as np
import scipy.io as sio

import ml.collaborative_filtering as cofi


def check_cost_function_with_preloaded_data(Y, R):
    num_users = 4
    num_movies = 5
    num_features = 3
    data = sio.loadmat("ml_course_material/machine-learning-ex8/ex8/ex8_movieParams.mat")
    X = data["X"][:num_movies, :num_features]
    Theta = data["Theta"][:num_users, :num_features]
    Y = Y[:num_movies, :num_users]
    R = R[:num_movies, :num_users]

    print(f"expected cost is 22.22, actual was {cofi.cost_function(X, Y, R, Theta)}")


if __name__ == "__main__":
    movies_data = sio.loadmat("ml_course_material/machine-learning-ex8/ex8/ex8_movies.mat")
    Y = movies_data["Y"]
    R = movies_data["R"]

    print(f"Toy story avg rating is {Y[0, np.nonzero(R[0, :])[0]].mean()}")

    check_cost_function_with_preloaded_data(Y, R)

