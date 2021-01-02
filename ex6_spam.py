import re

import numpy as np
import scipy.io as sio
from nltk.stem import PorterStemmer
from sklearn import svm


def preprocess_email_from_file(file_name):
    with open(file_name, "r") as file:
        return preprocess_email(file.read())


def preprocess_email(email):
    print(email)
    email = email.lower()
    email = re.sub("<[^<>]+>", " ", email)
    email = re.sub("(http|https)://[^\\s]*", "httpaddr", email)
    email = re.sub("[^\\s]+@[^\\s]+", "emailaddr", email)
    email = re.sub("[\\d]+", "number", email)
    email = re.sub("[$]+", "dollar", email)
    words = []
    for word in email.split():
        word_chars_only = re.sub("[^a-z]", "", word)
        if word_chars_only != "":
            words.append(word_chars_only)
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    print(stemmed_words)
    return stemmed_words


def load_vocabulary(file_name):
    with open(file_name, "r") as file:
        vocabulary = {}
        for line in file:
            words = line.split()
            assert len(words) == 2 and words[0].isnumeric() and words[1].isalpha(), "Unexpected line: " + line
            vocabulary[words[1]] = int(words[0])
        return vocabulary


def indices_into_features(vocabulary_values, word_indices):
    word_indices = set(word_indices)
    n = len(vocabulary)
    feature = np.empty(n)
    for i in range(n):
        feature[i] = 1 if vocabulary_values[i] in word_indices else 0
    return feature


def classify_spam(vocabulary):
    dataTrain = sio.loadmat("ml_course_material/machine-learning-ex6/ex6/spamTrain.mat")
    X_train = dataTrain["X"]
    y_train = dataTrain["y"]
    svc = svm.SVC(C=0.1, kernel='linear')
    svc.fit(X_train, y_train.flatten())

    predictions_train = svc.predict(X_train)
    print(f"train prediction accuracy: {np.mean(predictions_train == y_train.flatten())}")

    dataTest = sio.loadmat("ml_course_material/machine-learning-ex6/ex6/spamTest.mat")
    X_test = dataTest["Xtest"]
    y_test = dataTest["ytest"]

    predictions_test = svc.predict(X_test)
    print(f"test prediction accuracy: {np.mean(predictions_test == y_test.flatten())}")

    theta = svc.coef_

    max_indices = np.argsort(theta.flatten())[::-1]

    # -1 because in vocabulary it is not indexed from 0, but 1
    vocabulary_reversed = {index - 1: word for word, index in vocabulary.items()}

    print("most significant words:")
    for i in range(20):
        word = vocabulary_reversed[max_indices[i]]
        print(word)


if __name__ == "__main__":
    words = preprocess_email_from_file("ml_course_material/machine-learning-ex6/ex6/emailSample1.txt")
    vocabulary = load_vocabulary("ml_course_material/machine-learning-ex6/ex6/vocab.txt")
    indices = [vocabulary[word] for word in words if word in vocabulary]
    print(indices)
    features = indices_into_features(list(vocabulary.values()), indices)
    print(features)

    classify_spam(vocabulary)
