import datetime
import itertools
import json
#import tabular_data as td
import pandas as pd
import prepare_data
#import numpy as np
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

def run_baseline_SGDRegressor_model(airbnb_data):
    """
    Runs the SGDRegressor model using default parameters as a baseline model
    to compare the others to. No attempt is made to tune paramaters to improve
    performance.

    Args:
        airbnb_data:    tuple containing features and labels for training,
                        testing, and validation data sets

    Returns:
        None  
    
    """
    X_train_normalised, y_train, X_test_normalised, y_test, X_validation_normalised, y_validation = airbnb_data
    model = SGDRegressor()
    model.fit(X_train_normalised, y_train)

    y_pred_train = model.predict(X_train_normalised)
    y_pred_test = model.predict(X_test_normalised)
    rmse_train  = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print("Training Set Results:")
    print("RMSE:", rmse_train)
    print("R-squared:", r2_train)

    print("\nTest Set Results:")
    print("RMSE:", rmse_test)
    print("R-squared:", r2_test)

def generate_model_parameters():
    """
    This function creates a list of dictionaries containing the model specifications to be used
    by the custom_tune_regression_model_hyperparameters function.

    This was a sub-optimal solution to the problem which was superseded by the dictionary
    of dictionaries approach as contained in the generate_model_parameters_dictionary function.
    It is retained for interest and comparison purposes.

    Args:
        None

    Returns:
        model_specifications_list: a list of dictionaries each one of which contains the
                                   parameters for a specific regression model
    """
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

def custom_tune_regression_model_hyperparameters(airbnb_data):
    """
    Function to perform grid search over a range of parameters and regression models.
    This function does not use sklearn's built-in GridSearchCV function for the
    grid search but instead implements it by hand by iterating over a list containing
    all permutations of model specifications generated in the generate_model_parameters function.
    This is not the most elegant or efficient way to solve this problem!

    Args:
        X_train_normalised:         normalised training set features
        y_train:                    training set labels
        X_validation_normalised:    normalised validation set features
        y_validation:               validation set labels
        X_test_normalised:          normalised test set features
        y_test:                     test set labels 

    Returns:
        test_results:               a list containing model type, parameters, and results
                                    for each of the model configurations tested

    """
    X_train_normalised, y_train, X_test_normalised, y_test, X_validation_normalised, y_validation = airbnb_data
    model_specifications_list = generate_model_parameters()
    test_results = []

    for i in model_specifications_list:
        print("Running ", i)
        
        model_type = i[0]
        model_parameters = i[1]
    
        model = model_type(**model_parameters)

        model.fit(X_train_normalised, y_train)
        
        y_prediction_train = model.predict(X_train_normalised)
        train_RMSE = mean_squared_error(y_train, y_prediction_train, squared=False)
        r2_train = r2_score(y_train, y_prediction_train)

        y_prediction_validation = model.predict(X_validation_normalised)
        validation_RMSE = mean_squared_error(y_validation, y_prediction_validation, squared=False)
        r2_validation = r2_score(y_validation, y_prediction_validation)
        
        y_prediction_test = model.predict(X_test_normalised)
        sqrt_mse_test = mean_squared_error(y_test, y_prediction_test, squared=False)
        r2_test = r2_score(y_test, y_prediction_test)

        result = {"model": (model.__class__.__name__), "parameters": i[1], "sqrt_mse_validation": validation_RMSE,
                  "sqrt_mse_test": sqrt_mse_test, "r2_validation": r2_validation, "r2_test": r2_test}
        
        test_results.append(result)

    return test_results

def generate_model_parameters_dictionary():
    """
    This function creates the dictionary defining the search space for the
    tune_regression_model_hyperparameters_using_dict function. It gives a more
    elegant way to run the grid search than the previous function

    Args:
        None

    Returns:
        model_parameters_dictionary: a dictionary of dictionaries with each key being a regression
                                        model and each value being itself a dictionary defining a
                                        parameter space to be searched
    """

    model_parameters_dictionary = {SGDRegressor(): {"penalty": ["l1", "l2", "elasticnet"],
                                                    "loss": ["squared_error", "huber"],
                                                  "alpha": [0.01, 0.001, 0.0001, 0.00001],
                                                  "max_iter": [1000, 5000, 10000, 50000],
                                                  "eta0": [0.001, 0.01, 0.1]},

                                    DecisionTreeRegressor(): {"max_depth": [None, 2, 3, 5],"splitter": ["best"]},

                                    RandomForestRegressor(): {"n_estimators": [10, 100, 1000, 5000],
                                                            "max_depth": [None, 2, 10]},

                                    GradientBoostingRegressor(): {"learning_rate": [0.01, 0.1, 1],
                                                                "max_depth": [2, 3, 10],
                                                                "n_estimators": [10, 100, 1000]},

                                    SVR(): {"kernel": ["linear", "rbf", "sigmoid"],
                                          "C": [0.1, 1, 10]}

    }

    return model_parameters_dictionary

def tune_regression_model_hyperparameters_using_dict(airbnb_data):
    """
    Uses the dictionary of models and their parameter space returned by the
    generate_model_parameters_dictionary function as the search space for GridSearchCV.
    The best models and their parameters are returned by the function.

    Args:
        X_train
        y_train
        parameter_dictionary

    Returns:
        results_dictionary
    """
    X_train_normalised, y_train, X_test_normalised, y_test, X_validation_normalised, y_validation = airbnb_data
    parameter_dictionary = generate_model_parameters_dictionary()
    results_dictionary = {}
    for key in parameter_dictionary:
        print("Key: ", key)
        param_grid = parameter_dictionary[key]
        grid = GridSearchCV(key, param_grid, scoring="neg_root_mean_squared_error", verbose=1)
        grid.fit(X_train_normalised, y_train)
        results_df = pd.DataFrame(grid.cv_results_)
        print(results_df)
        print(grid.best_estimator_)
        print("Best params")
        print(grid.best_params_)
        for i in grid.best_params_:
            print(i)
        print(grid.best_score_)
        results_dictionary[grid.best_estimator_.__class__.__name__] = {"best params": grid.best_params_, "best score": grid.best_score_}
    
    print(results_dictionary)

    return results_dictionary

def save_regression_results_dictionary(results_dictionary):
    """
    Save the results_dictionary generated by the tune_regression_model_hyperparameters_using_dict function
    as a JSON file.

    Args:
        results_dictionary

    Returns:
        None
    """
    # Generate path name for results file
    output_dir = os.path.expanduser('~/Documents/AICore/Specialisation/Airbnb_Project/results/regression')
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    #Save results dictionary to a JSON file
    output_filename = f'regression_results_{current_datetime}.json'
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(results_dictionary, f, indent=4)

def regression_pipeline():
    """
    Imports the cleaned AirBnB dataset, and runs the SGDRegressor with default parameters
    before running a grid search over the data to explore several different regression models
    over a grid of parameters.

    Args:
        None

    Returns:
        None
    """
    print("Entering regression_pipeline")
    airbnb_data = prepare_data.prepare("Price_Night")
    #X_test_normalised, y_test, X_train_normalised, y_train, X_validation_normalised, y_validation
    print("Back in regression_pipeline")

    run_baseline_SGDRegressor_model(airbnb_data)
    save_regression_results_dictionary(tune_regression_model_hyperparameters_using_dict(airbnb_data))

if __name__ == "__main__":
    regression_pipeline()






















"""
    Old code calling previous versions of best regression model search. Retained for reference purposes and completeness 
    
    test_results = custom_tune_regression_model_hyperparameters(X_train_normalized, y_train, X_validation_normalized, y_validation, X_test_normalized, y_test)
    tune_regression_model_hyperparameters(X_train_normalized, y_train, X_validation_normalized, y_validation)
    best_model = find_best_model(test_results, False)
    print(best_model)
    save_test_results(test_results, best_model)
    """

"""
Old functions from previous versions of code retained here for sake of completeness

def save_model(best_model):
    
    # Generate path names for metric and configuration outputs 
    output_dir = os.path.expanduser('~/Documents/AICore/Specialisation/Airbnb_Project/results/regression')
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    #Save model specifications to a text file
    output_filename = f'regression_best_model_{current_datetime}.txt'
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:    
        json.dump(best_model, f, indent=4)

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

def find_best_model(test_results):
    # Function use MSE to determine best model
    best_metric = np.inf if use_MSE else -1

    for i in test_results:
        if i["validation_RMSE"] < best_metric:
            best_metric = i["validation_RMSE"]
            best_model = i

    return best_model

def tune_regression_model_hyperparameters(X_train, y_train, X_test, y_test):
    
    param_grid = {"C": [0.1, 1, 10, 100, 1000], "gamma": [1, 0.1, 0.01, 0.001, 0.0001], "kernel": ["rbf", "linear", "sigmoid"]}

    grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)
"""