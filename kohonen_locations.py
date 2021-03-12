import re
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np

import ml.kohonen as kh
import ml.bag_of_characters as boc


def preprocess(word):
    word = word.strip()
    word = re.sub("\s", "", word)
    word = re.sub("\"", "", word)
    word = re.sub("\)", "", word)
    word = re.sub("\(", "", word)
    word = re.sub("\.", "", word)
    word = re.sub(",", "", word)
    word = re.sub("-", "", word)
    word = re.sub("/", "", word)
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
    if iter in (50000, 90000):
        bmus = []
        map = np.empty((map_length, map_length), dtype=object)
        for i, x in enumerate(X):
            bmu_coords = np.unravel_index(np.argmin(np.linalg.norm(Thetas - x, axis=2)), (map_length, map_length))
            bmus.append(bmu_coords)
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
                    ax.text(neuron_x / map_length, neuron_y / map_length, str(len(items)) + " " +  "\n".join(items[:3]),
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
    with open("data/locations.csv") as locations_file:
        for loc in locations_file:
            locations.append(preprocess(loc))
    word_2_gramer = boc.WordNGramer(2)
    # locations = np.random.default_rng().permutation(locations)
    word_2_gramer.fit(locations)
    X = word_2_gramer.transform(locations)
        # np.save("locations", X)

    plt.ion()
    fig, ax = plt.subplots()
    map = kh.KohonenMap(100, learning_rate_constant=40000, init_sigma=20, sigma_constant=35000, observer=draw,
                        max_iter=120000)
    map.fit(X, locations)
    np.save("Thetas", map.Thetas)