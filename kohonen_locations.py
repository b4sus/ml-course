import datetime
import os.path
import re
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

import ml.kohonen as kh
import ml.bag_of_characters as boc


def preprocess(word):
    word = word.strip().lower()
    word = re.sub("\s", "", word)
    word = re.sub("\"", "", word)
    word = re.sub("\)", "", word)
    word = re.sub("\(", "", word)
    word = re.sub("\.", "", word)
    word = re.sub(",", "", word)
    word = re.sub("-", "", word)
    word = re.sub("/", "", word)
    word = re.sub("ä", "ae", word)
    word = re.sub("ö", "oe", word)
    word = re.sub("ü", "ue", word)
    word = re.sub("ß", "ss", word)
    return word




class Observer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.start = time.process_time()
        self.elapsed_times = []

    def __call__(self, X, map_length, iter, Thetas, learning_rate, sigma, X_repr, neighbour_change=-1):
        self.elapsed_times.append(time.process_time() - self.start)
        if len(self.elapsed_times) % 10 == 0:
            print(f"{datetime.datetime.now()}: iteration {iter}, learning_rate:{learning_rate}, sigma: {sigma}, "
                  f"neighbour_change: {neighbour_change} mean for one training sample: {np.mean(self.elapsed_times)}")
            self.elapsed_times.clear()
        if False and iter in (1000, 10000, 50000, 120000):
            map = np.empty((map_length, map_length), dtype=object)
            for i, x in enumerate(X):
                bmu_coords = np.unravel_index(np.argmin(np.linalg.norm(Thetas - x, axis=2)), (map_length, map_length))
                if map[bmu_coords[0], bmu_coords[1]] is None:
                    map[bmu_coords[0], bmu_coords[1]] = []
                map[bmu_coords[0], bmu_coords[1]].append(X_repr[i])

            plt.title(
                f"Iteration {iter} - learning rate {learning_rate:.3f}, sigma {sigma:.3f}, "
                + f"neighbour_change {neighbour_change:.3f}")
            for neuron_x in range(map_length):
                for neuron_y in range(map_length):
                    items = map[neuron_x, neuron_y]
                    if items:
                        self.ax.text(neuron_x / map_length, neuron_y / map_length, str(len(items)) + " " + "\n".join(items[:3]),
                                fontsize='xx-small')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.ax.clear()
        self.start = time.process_time()


def train():
    locations = load_preprocess_locations()
    word_2_gramer = boc.WordNGramer(2)

    map = kh.KohonenMap(70, learning_rate_constant=25_000, init_sigma=15, sigma_constant=15_000, observer=Observer(),
                        max_iter=100000)
    # map.fit(X, locations)
    pipeline = make_pipeline(word_2_gramer, MinMaxScaler(), map)
    pipeline.fit_transform(locations, kohonenmap__X_repr=locations)
    np.save(versioned_file_name("kohonen_locations_trained_thetas"), map.Thetas)


def load_preprocess_locations():
    locations = []
    with open("data/locations.csv", encoding="utf-8") as locations_file:
        for loc in locations_file:
            locations.append(preprocess(loc))
    return locations


def predict(word):
    locations = load_preprocess_locations()
    Thetas = np.load(versioned_file_name("kohonen_locations_trained_thetas.npy"))
    trained_map = kh.TrainedKohonenMap(Thetas)

    preprocess_pipeline = make_pipeline(boc.WordNGramer(1), MinMaxScaler(), trained_map)
    preprocess_pipeline.fit(locations)

    if os.path.exists(versioned_file_name("location_positions.npy")):
        location_positions = np.load(versioned_file_name("location_positions.npy"), allow_pickle=True)
    else:
        location_positions = preprocess_pipeline.transform(locations)
        np.save(versioned_file_name("location_positions"), location_positions)

    map_length = len(Thetas)
    coords = np.empty((map_length, map_length, 2))
    for neuron_x in range(map_length):
        for neuron_y in range(map_length):
            coords[neuron_x, neuron_y] = np.array([neuron_x, neuron_y])

    map = np.empty((map_length, map_length), dtype=object)
    for location, position in zip(locations, location_positions):
        x, y = position
        if map[x, y] is None:
            map[x, y] = []
        map[x, y].append(location)

    transformed_samples = preprocess_pipeline.transform([word])

    generator = generate_closest(transformed_samples[0], coords, map)

    for i in range(30):
        print(next(generator))


def generate_closest(sample, coords, location_map):
    start = time.process_time()
    D = np.linalg.norm(sample - coords, axis=2)
    xs, ys = np.unravel_index(np.argsort(D, axis=None), D.shape)
    print(time.process_time() - start)
    for x, y in zip(xs, ys):
        if location_map[x, y] is not None:
            for loc in location_map[x, y]:
                yield loc


def versioned_file_name(file_name):
    if "." in file_name:
        base_name, suffix = file_name.rsplit(".", maxsplit=1)
        return base_name + version + "." + suffix
    else:
        return file_name + version


if __name__ == "__main__":
    version = "1gram"
    if not os.path.exists(versioned_file_name("kohonen_locations_trained_thetas.npy")):
        train()
    predict("hechngeli")
