import datetime
import itertools
import json
import tabular_data as td
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

def main():
        
    df = pd.read_csv("tabular_data/listing.csv")
    df = td.clean_tabular_data(df)
    df = df.astype({"guests": "int32", "bedrooms": "int32"})

#    X, y = td.load_airbnb(df, "Price_Night")
    X, y = td.load_airbnb(df, "beds")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=47)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5, random_state=37)

    scaler = preprocessing.RobustScaler()
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    X_validation_normalized = scaler.transform(X_validation)    

    """
    normalizer = preprocessing.MinMaxScaler().fit(X_train)
    X_train_normalized = normalizer.transform(X_train)
    X_test_normalized = normalizer.transform(X_test)
    X_validation_normalized = normalizer.transform(X_validation)
    """

    test_results = custom_tune_regression_model_hyperparameters(X_train_normalized, y_train, X_validation_normalized, y_validation, X_test_normalized, y_test)
    #tune_regression_model_hyperparameters(X_train_normalized, y_train, X_validation_normalized, y_validation)
    best_model = find_best_model(test_results, False)
    print(best_model)
    save_test_results(test_results, best_model)

def tune_regression_model_hyperparameters(X_train, y_train, X_test, y_test):
    
    param_grid = {"C": [0.1, 1, 10, 100, 1000], "gamma": [1, 0.1, 0.01, 0.001, 0.0001], "kernel": ["rbf", "linear", "sigmoid"]}

    grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)

def generate_model_parameters():

    model_specifications  = [
                            
                            [SGDRegressor,
                            ["penalty", "alpha", "max_iter", "eta0", "random_state"],
                            [["l1", "l2", "elasticnet"], [0.01, 0.001, 0.0001, 0.00001], [100, 500, 1000, 10000, 50000, 100000], [0.001, 0.01, 0.1]], [47]],

                            [DecisionTreeRegressor,
                            ["max_depth", "splitter"],
                            [[None, 2, 3, 5], ["best"]]],

                            [RandomForestRegressor,
                            ["n_estimators", "max_depth"],
                            [[10, 100, 1000, 5000], [None, 2, 10]]],

                            [GradientBoostingRegressor, 
                            ["learning_rate", "max_depth", "n_estimators"], 
                            [[0.01, 0.1, 1], [2, 3, 10], [10, 100, 1000]]],

                            [SVR,
                            ["kernel", "C"],
                            [["linear", "rbf", "sigmoid"], [0.1, 1, 10]]],

                            ]

    model_specifications_list = []
    for i in model_specifications:
        parameter_values = list(itertools.product(*i[2]))
        parameter_keys = i[1]
        for j in parameter_values:
            model_specifications_list.append((i[0], dict(zip(parameter_keys, j))))

    return model_specifications_list

def save_model(best_model):
    
    # Generate path names for metric and configuration outputs 
    output_dir = os.path.expanduser('~/Documents/AICore/Specialisation/Airbnb_Project/results/regression')
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    #Save model specifications to a text file
    output_filename = f'regression_best_model_{current_datetime}.txt'
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:    
        json.dump(best_model, f, indent=4)

def custom_tune_regression_model_hyperparameters(X_train_normalized, y_train, X_validation_normalized, y_validation, X_test_normalized, y_test):
    
    model_specifications_list = generate_model_parameters()
    test_results = []

    for i in model_specifications_list:
        print("Running ", i)
        
        model_type = i[0]
        model_parameters = i[1]
    
        model = model_type(**model_parameters)

        model.fit(X_train_normalized, y_train)
        
        y_prediction_validation = model.predict(X_validation_normalized)
        sqrt_mse_validation = mean_squared_error(y_validation, y_prediction_validation, squared=False)
        r2_validation = r2_score(y_validation, y_prediction_validation)
        
        y_prediction_test = model.predict(X_test_normalized)
        sqrt_mse_test = mean_squared_error(y_test, y_prediction_test, squared=False)
        r2_test = r2_score(y_test, y_prediction_test)

        result = {"model": (model.__class__.__name__), "parameters": i[1], "sqrt_mse_validation": sqrt_mse_validation,
                  "sqrt_mse_test": sqrt_mse_test, "r2_validation": r2_validation, "r2_test": r2_test}
        
        test_results.append(result)

    return test_results

def save_test_results(test_results, best_model):

    # Generate path names for metric and configuration outputs 
    output_dir = os.path.expanduser('~/Documents/AICore/Specialisation/Airbnb_Project/results/regression')
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    #Save regression results to a text file
    output_filename = f'regression_results_{current_datetime}.json'
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    
    #Save model specifications to a text file
    output_filename = f'regression_best_model_{current_datetime}.json'
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:    
        json.dump(best_model, f, indent=4)

def find_best_model(test_results, use_MSE=True):
    # Function use MSE or r-squared to determine best model
    best_metric = np.inf if use_MSE else -1

    for i in test_results:
        if use_MSE:
            if i["sqrt_mse_test"] < best_metric:

                best_metric = i["sqrt_mse_test"]
                best_model = i
        
        if not(use_MSE):
            if i["r2_test"] > best_metric:
                best_metric = i["r2_test"]
                best_model = i

    return best_model

if __name__ == "__main__":
    main()