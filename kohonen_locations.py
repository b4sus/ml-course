import re
import time
import datetime

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


start = time.process_time()
elapsed_times = []


def draw(X, map_length, iter, Thetas, learning_rate, sigma, X_repr, neighbour_change=-1):
    global elapsed_times, start
    elapsed_times.append(time.process_time() - start)
    if len(elapsed_times) % 10 == 0:
        print(f"{datetime.datetime.now()}: iteration {iter}, learning_rate:{learning_rate}, sigma: {sigma}, "
              f"neighbour_change: {neighbour_change} mean for one training sample: {np.mean(elapsed_times)}")
        elapsed_times.clear()
    if iter in (1000, 10000, 50000, 120000):
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
                    ax.text(neuron_x / map_length, neuron_y / map_length, str(len(items)) + " " + "\n".join(items[:3]),
                            fontsize='xx-small')
        fig.canvas.draw()
        fig.canvas.flush_events()
        ax.clear()
    start = time.process_time()


if __name__ == "__main__":
    # if os.path.exists("locations.npy"):
    #     X = np.load("locations.npy")
    # else:
    locations = []
    with open("data/locations.csv", encoding="utf-8") as locations_file:
        for loc in locations_file:
            locations.append(preprocess(loc))
    word_2_gramer = boc.WordNGramer(2)
    # locations = np.random.default_rng().permutation(locations)
    # word_2_gramer.fit(locations)
    # print(f"corpus len {len(word_2_gramer.corpus)}")
    # print(word_2_gramer.corpus)
    # X = word_2_gramer.transform(locations)

    # np.save("locations", X)

    plt.ion()
    fig, ax = plt.subplots()
    map = kh.KohonenMap(100, learning_rate_constant=50000, init_sigma=20, sigma_constant=35000, observer=draw,
                        max_iter=140000)
    # map.fit(X, locations)
    pipeline = make_pipeline(word_2_gramer, MinMaxScaler(), map)
    pipeline.fit_transform(locations, kohonenmap__X_repr=locations)
    np.save("Thetas", map.Thetas)
