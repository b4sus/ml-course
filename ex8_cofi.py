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
    cofi.cost_function_derivative(X, Y, R, Theta)

    print(f"expected regularized cost is 31.34, actual was {cofi.cost_function(X, Y, R, Theta, 1.5)}")


def mean_normalization(Y, R):
    # movie_means = (Y * R).mean(1).reshape((-1, 1))
    # Movie_means = movie_means.repeat(Y.shape[1], 1)
    # return Y - (Movie_means * R), movie_means

    movie_means = np.empty((Y.shape[0], 1))
    Y_norm = np.empty(Y.shape)
    for i in range(len(Y)):
        movie_means[i, 0] = Y[i, np.nonzero(R[i, :])[0]].mean()
        Y_norm[i, :] = Y[i, :] - movie_means[i, 0]
    return Y_norm, movie_means


if __name__ == "__main__":
    movies_data = sio.loadmat("ml_course_material/machine-learning-ex8/ex8/ex8_movies.mat")
    Y = movies_data["Y"]
    R = movies_data["R"]

    print(f"Toy story avg rating is {Y[0, np.nonzero(R[0, :])[0]].mean()}")

    check_cost_function_with_preloaded_data(Y, R)

    movies = []
    with open("ml_course_material/machine-learning-ex8/ex8/movie_ids.txt", encoding="ISO-8859-1") as movie_ids:
        for line in movie_ids.readlines():
            line_elements = line.split(maxsplit=1)
            movies.append(line_elements[1].rstrip())

    my_ratings = np.zeros(len(movies))
    my_ratings[55] = 5
    my_ratings[66] = 5
    my_ratings[68] = 5
    my_ratings[72] = 5
    my_ratings[77] = 1
    my_ratings[111] = 1
    my_ratings[150] = 1
    my_ratings[150] = 1

    Y = np.hstack((my_ratings.reshape((-1, 1)), Y))
    R = np.hstack((my_ratings.reshape((-1, 1)) != 0, R))

    Y_norm, movie_means = mean_normalization(Y, R)

    X, Theta = cofi.learn(Y_norm, R, 10, 1.5)

    my_predictions = X @ Theta[0, :].T

    for i, movie in enumerate(movies):
        print(f"Prediction for movie {movie} is {my_predictions[i] + movie_means[i, 0]}")





