import scipy.io as sio

if __name__ == "__main__":
    movies_data = sio.loadmat("ml_course_solutions/machine-learning-ex8/ex8/ex8_movies.mat")
    Y = movies_data["Y"]
    R = movies_data["R"]