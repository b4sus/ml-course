import csv
import numpy as np
import ml.ml as ml
import ml.predict as predict
import ml.feature as feature
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

# passengers = [p for p in passengers if p.fare < 500] really huge (well above average)
# entries can cause division by 0 when they are multiplied by polynomial features

X = np.empty((0, 4))
y = np.empty((len(passengers), 1))

idx = 0
for passenger in passengers:
    X = np.vstack((X, [passenger.sex, passenger.age, passenger.ticket_class, passenger.fare]))
    y[idx, 0] = 1.0 if passenger.survived else 0.0
    idx = idx + 1

plt.plot()

print("Vanilla X({} {}):\n{}".format(*X.shape, X))

X = feature.one_hot_encode(X, 2)

print("Ticket encoded X({} {}):\n{}".format(*X.shape, X))

polynomial_features = PolynomialFeatures(2, include_bias=False, interaction_only=True)
X = polynomial_features.fit_transform(X)
print(polynomial_features.get_feature_names())
print("Polynomial X({} {}):\n{}".format(*X.shape, X))

X = feature.reduce_features_without_std(X)
print("Reduced features X({} {}):\n{}".format(*X.shape, X))

normalizer = feature.FeatureNormalizer(X)
X = normalizer.normalized_x_m
X = np.hstack((np.ones((X.shape[0], 1)), X))
# X = Normalizer().fit_transform(X)

print("Final X({} {}):\n{}".format(*X.shape, X))

# (theta, costs) = ml.gradient_descent(X, y, ml.logistic_regression_cost, ml.logistic_regression_cost_derivative)

(theta, num_of_evaluations, return_code) = op.fmin_tnc(func=ml.logistic_regression_cost_gradient,
                                                       x0=np.zeros((X.shape[1], 1)),
                                                       args=(X, y, 0))
print("Return code: {}".format(return_code))

theta = theta.reshape((X.shape[1], 1))
predictions = predict.predict(X[:, 1:], theta, ml.logistic_regression_hypothesis)

print(np.mean(predictions == y))
