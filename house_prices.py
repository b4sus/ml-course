import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as op
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn import svm

import ml.linear_regression as lire


def learn_manually_with_scipy(X, y, regularization_lambda):
    op_res = op.minimize(fun=lire.linear_regression_cost_gradient,
                         x0=np.zeros((X.shape[1])),
                         args=(X, y, regularization_lambda),
                         method="CG",
                         jac=True)

    learned_theta = op_res.x

    return learned_theta.reshape((-1, 1))


def learn_with_sklearn(X, y):
    # return LinearRegression().fit(X, y)
    return Ridge().fit(X, y)
    # return RandomForestClassifier(max_features=10).fit(X, y.reshape(-1))
    # return svm.SVR(kernel="linear").fit(X, y.reshape(-1))


def predict_test_houses(pipeline, estimator):
    test_houses = pd.read_csv("house_prices/test.csv")

    test_ids = pd.DataFrame(test_houses["Id"]).set_index("Id")

    X_real_test = pipeline.transform(test_houses.drop("Id", axis=1))
    X_real_test = np.hstack((np.ones((X_real_test.shape[0], 1)), X_real_test))

    y_real_pred = estimator.predict(X_real_test)

    test_ids["SalePrice"] = y_real_pred

    test_ids.to_csv("house_prices/predictions.csv")


def prepare_pipeline():
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    quality_categories = ["NA", "Po", "Fa", "TA", "Gd", "Ex"]

    full_pipeline = make_column_transformer(
        (num_pipeline, list(train_houses_numeric_only)),
        (OneHotEncoder(),
         ["LotShape", "LandContour", "Neighborhood", "Condition1", "BldgType", "HouseStyle", "Foundation"]),
        (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder()),
         ["BsmtQual", "BsmtCond", "BsmtExposure", "KitchenQual", "Functional", "SaleType"]),
        (make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder()),
         ["GarageType", "GarageFinish", "Fence", "BsmtExposure",
          "SaleCondition"]),
        (make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder(categories=[quality_categories]),
                       StandardScaler()), ["KitchenQual"]),
        (make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                       OrdinalEncoder(
                           categories=[quality_categories, quality_categories, quality_categories, quality_categories,
                                       ]), StandardScaler()),
         ["FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]),
        (OrdinalEncoder(categories=[quality_categories, quality_categories]), ["ExterQual", "ExterCond"])
    )

    return full_pipeline


def print_cv_scores(scores):
    print(f"Scores: {scores}")
    print(f"Mean: {scores.mean()}")
    print(f"Std: {scores.std()}")


def grid_search_ridge(X, y):
    grid_search = GridSearchCV(Ridge(), [{"alpha": [0.1, 0.3, 1, 3, 9, 11, 12, 13, 14, 15, 27, 100]}], cv=5,
                               scoring="neg_root_mean_squared_error",
                               return_train_score=True)
    grid_search.fit(X, y.reshape(-1))
    print(f"Best estimator {grid_search.best_estimator_}")
    print(f"Best score {-grid_search.best_score_}")
    return grid_search.best_estimator_


def grid_search_random_forest(X, y):
    param_grid = [
    {"n_estimators": [20, 30, 90, 110, 120], "max_features": [1, 3, 9, 15, 27]}
    ]

    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="neg_root_mean_squared_error", return_train_score=True)
    grid_search.fit(X, y.reshape(-1))
    print(f"Best estimator {grid_search.best_estimator_}")
    print(f"Best score {-grid_search.best_score_}")
    return grid_search.best_estimator_


if __name__ == "__main__":
    houses = pd.read_csv("house_prices/train.csv")
    prices = houses["SalePrice"]
    houses = houses.drop(["Id", "SalePrice", "MoSold", "3SsnPorch", "BsmtFinSF2", "BsmtHalfBath", "MiscVal"], axis=1)

    # trainset.hist()
    # plt.show()

    split = StratifiedShuffleSplit(1, test_size=0.2, random_state=53)
    for train_index, test_index in split.split(houses, houses["OverallQual"]):
        train_houses = houses.iloc[train_index]
        train_prices = prices.iloc[train_index]
        test_houses = houses.iloc[test_index]
        test_prices = prices.iloc[test_index]
    # (train_houses, test_houses, train_prices, test_prices) = train_test_split(houses, prices, test_size=0.2,
    #                                                                           random_state=53)

    train_houses_numeric_only = train_houses.select_dtypes(include=np.number)

    print(train_houses_numeric_only.head())

    full_pipeline = prepare_pipeline()

    train_transformed = full_pipeline.fit_transform(train_houses)

    y = train_prices.to_numpy().reshape((-1, 1))
    X = train_transformed
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    theta_scipy = learn_manually_with_scipy(X, y, 1)

    print(f"training set RMSE from learning manually is {lire.rmse(theta_scipy, X, y)}")

    # scores = cross_val_score(RandomForestClassifier(max_features=10), X, y.reshape(-1),
    #                          scoring="neg_mean_squared_error", cv=10)
    # print_cv_scores(np.sqrt(-scores))

    best_ridge_estimator = grid_search_random_forest(X, y)

    y_test = test_prices.to_numpy().reshape((-1, 1))
    X_test = full_pipeline.transform(test_houses)
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    print(f"test set RMSE from learning manually is {lire.rmse(theta_scipy, X_test, y_test)}")

    # regr = learn_with_sklearn(X, y)
    regr = best_ridge_estimator
    print(
        f"training set RMSE from learning with sklearn is {mean_squared_error(y, regr.predict(X), squared=False)}")
    print(
        f"test set RMSE from learning with sklearn is {mean_squared_error(y_test, regr.predict(X_test), squared=False)}")

    # predict_test_houses(full_pipeline, regr)
