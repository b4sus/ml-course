import csv
import random
import numpy as np
import ml.ml as ml
import ml.predict as predict
import ml.feature as feature
import ml.pipeline as pipeline
import scipy.optimize as op
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

np.seterr(all="raise")


class Passenger:
    def __init__(self, param_dict):
        self.id = param_dict['PassengerId']
        self.name = param_dict['Name']
        self.survived = bool(int(param_dict['Survived']))
        if param_dict['Age']:
            self.age = float(param_dict['Age'])
        else:
            self.age = float(40)

        self.ticket_class = int(param_dict['Pclass'])
        self.sex = float(-1) if param_dict['Sex'] == 'male' else float(1)
        self.num_siblings_spouses = int(param_dict['SibSp'])
        self.num_parents_children = int(param_dict['Parch'])
        self.ticket_nr = param_dict['Ticket']
        self.fare = float(param_dict['Fare'])
        self.cabin = param_dict['Cabin']
        self.embarked = param_dict['Embarked']

    def __str__(self):
        return f'{self.id} - {self.name} - {self.survived}'


passengers = []

with open('data/titanic-train.csv', 'r') as train_csv:
    dict_reader = csv.DictReader(train_csv)
    for row in dict_reader:
        passengers.append(Passenger(row))

# random.shuffle(passengers)

train_passengers = passengers[:int(len(passengers) * 0.8)]
test_passengers = passengers[int(len(passengers) * 0.8):]

# passengers = [p for p in passengers if p.fare < 500] really huge (well above average)
# entries can cause division by 0 when they are multiplied by polynomial features

def create_feature_matrix(passengers):
    X = np.empty((0, 4))
    y = np.empty((len(passengers), 1))
    for (idx, passenger) in enumerate(passengers):
        X = np.vstack((X, [passenger.sex, passenger.age, passenger.ticket_class, passenger.fare]))
        y[idx, 0] = 1.0 if passenger.survived else 0.0
    return X, y


(X_train, y) = create_feature_matrix(train_passengers)

pipeline = pipeline.Pipeline()
pipeline.one_hot_encode(2)
pipeline.polynomial(2, include_bias=False, interaction_only=True)
pipeline.reduce_features_without_std()
pipeline.normalize()
pipeline.bias()

(theta_from_pipeline, X_processed) = pipeline.execute_train(X_train, y)

predictions = predict.predict(X_processed[:, 1:], theta_from_pipeline, ml.logistic_regression_hypothesis)

print(np.mean(predictions == y))

(X_test, y_test) = create_feature_matrix(test_passengers)
X_test = pipeline.process_test(X_test)

print("X_test({} {}):\n{}".format(*X_test.shape, X_test))
