import os.path
import re

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
    map = kh.KohonenMap(100, len(word_2_gramer.corpus))
    map.fit(X)