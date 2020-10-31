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
        if param_dict['Age']:
            self.age = float(param_dict['Age'])
        else:
            self.age = float(40)

        self.ticket_class = int(param_dict['Pclass'])
        self.sex = float(-1) if param_dict['Sex'] == 'male' else float(1)
        self.num_siblings_spouses = int(param_dict['SibSp'])
        self.num_parents_children = int(param_dict['Parch'])
        self.ticket_nr = param_dict['Ticket']
        self.fare = float(param_dict['Fare']) if param_dict['Fare'] else 15.
        self.cabin = param_dict['Cabin']
        self.embarked = param_dict['Embarked']

    def __str__(self):
        return f'{self.id} - {self.name} - {self.survived}'


class TrainPassenger(Passenger):
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.survived = bool(int(param_dict['Survived']))

    def __str__(self):
        return f"{self.id} - {self.name}"


passengers = []

with open('data/titanic-train.csv', 'r') as train_csv:
    dict_reader = csv.DictReader(train_csv)
    for row in dict_reader:
        passengers.append(TrainPassenger(row))

# random.shuffle(passengers)

# passengers = [p for p in passengers if p.fare < 500]  # really huge (well above average)
# entries can cause division by 0 when they are multiplied by polynomial features

train_passengers = passengers[:int(len(passengers) * 0.9)]
validation_passengers = passengers[int(len(passengers) * 0.9):]


def create_feature_matrix(passengers):
    X = np.empty((0, 6))
    for (idx, passenger) in enumerate(passengers):
        X = np.vstack((X, [passenger.sex, passenger.age, passenger.ticket_class, passenger.fare,
                           passenger.num_siblings_spouses, passenger.num_parents_children]))
    return X


def create_result_vector(passengers):
    y = np.empty((len(passengers), 1))
    for (idx, passenger) in enumerate(passengers):
        y[idx, 0] = 1.0 if passenger.survived else 0.0
    return y


X_train = create_feature_matrix(train_passengers)
y = create_result_vector(train_passengers)

pipeline = pipeline.Pipeline()
pipeline.one_hot_encode(2)
pipeline.polynomial(3, include_bias=False, interaction_only=True)
pipeline.reduce_features_without_std()
pipeline.normalize()
pipeline.bias()

(theta_from_pipeline, X_processed) = pipeline.execute_train(X_train, y, regularization=1)

predictions = predict.predict(X_processed[:, 1:], theta_from_pipeline, ml.logistic_regression_hypothesis)

print(np.mean(predictions == y))

X_validation = create_feature_matrix(validation_passengers)
y_validation = create_result_vector(validation_passengers)

X_validation = pipeline.process_test(X_validation)

predictions_validation = predict.predict(X_validation[:, 1:], theta_from_pipeline, ml.logistic_regression_hypothesis)

print(np.mean(predictions_validation == y_validation))

test_passengers = []

with open('data/titanic-test.csv', 'r') as test_csv:
    dict_reader = csv.DictReader(test_csv)
    for row in dict_reader:
        test_passengers.append(Passenger(row))

X_test = create_feature_matrix(test_passengers)

X_test = pipeline.process_test(X_test)

predictions_test = predict.predict(X_test[:, 1:], theta_from_pipeline, ml.logistic_regression_hypothesis)

# print(predictions_test)

results = zip([p.id for p in test_passengers], predictions_test.flatten())

# for zipped in results:
#     print(zipped)

with open("titanic-result.csv", "w", newline='') as result_csv:
    writer = csv.writer(result_csv, delimiter=",")
    writer.writerow(("PassengerId", "Survived"))
    for result in results:
        writer.writerow(result)
