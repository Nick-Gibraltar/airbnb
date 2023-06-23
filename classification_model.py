import itertools
import tabular_data as td
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

def generate_model_parameters():

    model_specifications  = [
                            
                            [SGDClassifier,
                            ["penalty", "alpha", "max_iter", "learning_rate", "eta0", "random_state"],
                            [["l1", "l2", "elasticnet"], [0.01, 0.001, 0.0001, 0.00001], [100, 500, 1000, 10000, 50000, 100000], ["constant", "optimal", "adaptive"], [0.001, 0.01, 0.1]], [47]],
                            
                            [LogisticRegression,
                            [],
                            []],

                            [DecisionTreeClassifier,
                            ["max_depth", "splitter"],
                            [[None, 2, 3, 5, 10], ["best"]]],

                            [RandomForestClassifier,
                            ["n_estimators", "max_depth"],
                            [[10, 100, 1000, 5000], [None, 2, 10]]],

                            [GradientBoostingClassifier, 
                            ["learning_rate", "n_estimators", "max_depth"], 
                            [[0.01, 0.1, 1], [10, 100, 1000], [None, 2, 3, 5, 10]]],

                            [SVC,
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

def custom_tune_classification_model_hyperparameters(X_train_normalized, y_train, X_validation_normalized, y_validation, X_test_normalized, y_test):
    
    model_specifications_list = generate_model_parameters()
    test_results = []
    #min_sqrt_mse = np.inf
    #best_r2 = -1

    best_accuracy = 0
    for i in model_specifications_list:
        #print("Running ", i)

        model_type = i[0]
        model_parameters = i[1]
    
        model = model_type(**model_parameters)

        model.fit(X_train_normalized, y_train)
        
        y_prediction_validation = model.predict(X_validation_normalized)
        accuracy_validation = accuracy_score(y_validation, y_prediction_validation)
                
        y_prediction_test = model.predict(X_test_normalized)
        accuracy_test = accuracy_score(y_test, y_prediction_test)

        if accuracy_validation > best_accuracy:
            print("New best: ", accuracy_validation, "Test accuracy: ", accuracy_test, i)
            best_accuracy = accuracy_validation

        test_results.append([i[0], i[1], accuracy_validation, accuracy_test])

    #save_model(best_model)
    return test_results


def main():
        
    df = pd.read_csv("tabular_data/listing.csv")
    df = td.clean_tabular_data(df)
    df = df.astype({"guests": "int32", "bedrooms": "int32"})

    X, y = td.load_airbnb(df, "Category")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=47)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5, random_state=37)

    scaler = preprocessing.RobustScaler()
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    X_validation_normalized = scaler.transform(X_validation)

    test_results = custom_tune_classification_model_hyperparameters(X_train_normalized, y_train, X_validation_normalized, y_validation, X_test_normalized, y_test)
    #best_model = find_best_model(test_results, False)
    print(best_model)

    """
    logreg = LogisticRegression()
    logreg.fit(X_train_normalized, y_train)
    y_pred = logreg.predict(X_test_normalized)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    """

if __name__ == "__main__":
    main()