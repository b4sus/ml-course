import csv
import numpy as np
import ml.ml as ml

class Passenger:
    def __init__(self, param_dict):
        self.id = param_dict['PassengerId']
        self.name = param_dict['Name']
        self.survived = bool(int(param_dict['Survived']))
        if param_dict['Age']:
            self.age = float(param_dict['Age'])
        else:
            self.age = float(40)

        self.ticket_class = param_dict['Pclass']
        self.sex = float(-1) if param_dict['Sex'] == 'male' else float(1)
        self.num_siblings_spouses = int(param_dict['SibSp'])
        self.num_parents = int(param_dict['Parch'])
        self.ticket_nr = param_dict['Ticket']
        self.fare = float(param_dict['Fare'])
        self.cabin = param_dict['Cabin']
        self.embarked = param_dict['Embarked']

    def __str__(self):
        return f'{self.id} - {self.name} - {self.survived}'


passengers = []

with open('../train.csv', 'r') as train_csv:
    dict_reader = csv.DictReader(train_csv)
    for row in dict_reader:
        passengers.append(Passenger(row))

X = np.ones((len(passengers), 3))  # sex and age
y = np.empty(len(passengers))

idx = 0
for passenger in passengers:
    X[idx, 1:] = [passenger.sex, passenger.age]
    y[idx] = 1.0 if passenger.survived else 0.0
    idx = idx + 1

print(X, y)

theta = ml.gradient_descent(X, y.T, ml.logistic_regression_cost, ml.logistic_regression_cost_derivative)
