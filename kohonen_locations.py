import re

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


def draw(X, map_length, iter, Thetas, learning_rate, sigma, X_repr, neighbour_change=-1):
    if iter % 10 != 0:
        return
    bmus = []
    map = np.empty((map_length, map_length), dtype=object)
    for i, x in enumerate(X):
        bmu_score = np.inf
        # bmu_coords = None
        bmu_coords = np.unravel_index(np.argmin(np.linalg.norm(Thetas - x, axis=2)), (100, 100))
        # for neuron_x in range(map_length):
        #     for neuron_y in range(map_length):
        #         theta = Thetas[neuron_x, neuron_y]
        #         score = np.linalg.norm(x - theta)
        #         if score < bmu_score:
        #             bmu_coords = (neuron_x, neuron_y)
        #             bmu_score = score
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

if __name__ == "__main__":
    # if os.path.exists("locations.npy"):
    #     X = np.load("locations.npy")
    # else:
    locations = []
    with open("data/locations.csv") as locations_file:
        for loc in locations_file:
            locations.append(preprocess(loc))
    word_2_gramer = boc.WordNGramer(2)
    word_2_gramer.fit(locations)
    X = word_2_gramer.transform(locations)
        # np.save("locations", X)

    plt.ion()
    fig, ax = plt.subplots()
    map = kh.KohonenMap(100, learning_rate_constant=5000, init_sigma=25, sigma_constant=5000, observer=draw)
    map.fit(X, locations)
    np.save("Thetas", kh.Thetas)