import re

import numpy as np


class WordNGramer:
    def __init__(self, n):
        self.n = n

    def fit(self, words):
        corpus = set()
        for word in words:
            for n_gram in self.split(word.lower()):
                corpus.add(n_gram)
        self.corpus = list(sorted(corpus))

    def transform(self, words):
        X = np.zeros((len(words), len(self.corpus)), dtype=np.int32)
        for word_idx, word in enumerate(words):
            for n_gram_idx, n_gram in enumerate(self.corpus):
                if n_gram in word.lower():
                    X[word_idx, n_gram_idx] += 1
        return X

    def split(self, word):
        for i in range(0, (len(word) - len(word) % self.n) - 1):
            yield word[i:i + self.n]


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
    locations = []
    with open("../data/locations.csv") as locations_file:
        for loc in locations_file:
            locations.append(preprocess(loc))

    locations = [loc for loc in locations if len(loc) > 1]
    word_2_gramer = WordNGramer(2)
    word_2_gramer.fit(locations)
    # word_2_gramer.fit(["aa "])
    print(word_2_gramer.corpus)


    X = word_2_gramer.transform(locations[:10])
    indices = np.argwhere(X[2, :] > 0).ravel()
    print(locations[2])
    for i in indices:
        print(word_2_gramer.corpus[i])

