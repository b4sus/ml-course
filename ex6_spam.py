import re

import scipy.io as sio
from nltk.stem import PorterStemmer


def preprocess_email_from_file(file_name):
    file = open(file_name, "r")
    preprocess_email(file.read())


def preprocess_email(email):
    print(email)
    email = email.lower()
    email = re.sub("<[^<>]+>", " ", email)
    email = re.sub("(http|https)://[^\\s]*", "httpaddr", email)
    email = re.sub("[^\\s]+@[^\\s]+", "emailaddr", email)
    email = re.sub("[\\d]+", "number", email)
    email = re.sub("[$]+", "dollar", email)
    words = email.split()
    words = []
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    print(stemmed_words)


def classify_spam():
    data = sio.loadmat("ml_course_solutions/machine-learning-ex6/ex6/spamTrain.mat")
    print(data)


if __name__ == "__main__":
    preprocess_email_from_file("ml_course_solutions/machine-learning-ex6/ex6/emailSample1.txt")