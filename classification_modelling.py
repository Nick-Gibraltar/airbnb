import datetime
import itertools
import json
import os
import pandas as pd
import prepare_data
import tabular_data as td

from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_baseline_LogisticRegression_model()
def generate_model_parameters_dictionary():
    
    model_parameters_dictionary = {SGDClassifier(): {"penalty": ["l1", "l2", "elasticnet"],
                                                  "alpha": [0.01, 0.001, 0.0001, 0.00001],
                                                  "max_iter": [1000, 10000, 50000, 100000],
                                                  "learning_rate": []
                                                  "eta0": [0.001, 0.01, 0.1],
                                                  },

                                    DecisionTreeRegressor(): {"max_depth": [None, 2, 3, 5],"splitter": ["best"]},

                                    RandomForestRegressor(): {"n_estimators": [10, 100, 1000, 5000],
                                                            "max_depth": [None, 2, 10]},

                                    GradientBoostingRegressor(): {"learning_rate": [0.01, 0.1, 1],
                                                                "max_depth": [2, 3, 10],
                                                                "n_estimators": [10, 100, 1000]},

                                    SVR(): {"kernel": ["linear", "rbf", "sigmoid"],
                                          "C": [0.1, 1, 10]}

    }


def generate_model_parameters():

    model_specifications  = [
                            
                            [SGDClassifier,
                            ["penalty", "alpha", "max_iter", "learning_rate", "eta0", "random_state"],
                            [["l1", "l2", "elasticnet"], [0.01, 0.001, 0.0001, 0.00001], [100, 500, 1000, 10000, 50000, 100000],
                              ["constant", "optimal", "adaptive"], [0.001, 0.01, 0.1]], [47]],
                            
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

def save_test_results(test_results, best_model):

    # Generate path names for metric and configuration outputs 
    output_dir = os.path.expanduser('~/Documents/AICore/Specialisation/Airbnb_Project/results/classification')
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    #Save model specifications to a text file
    output_filename = f'classification_results_{current_datetime}.txt'
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        for i in test_results:
            f.write(i)
    
    #Save best model to a text file
    output_filename = f'classification_best_model_{current_datetime}.json'
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(best_model, f, indent=4)

def custom_tune_classification_model_hyperparameters(X_train_normalized, y_train, X_validation_normalized, y_validation, X_test_normalized, y_test):
    
    model_specifications_list = generate_model_parameters()
    test_results = []


    best_accuracy = 0
    for i in model_specifications_list:
    
        model_type = i[0]
        model_parameters = i[1]
    
        model = model_type(**model_parameters)

        model.fit(X_train_normalized, y_train)
        
        y_prediction_validation = model.predict(X_validation_normalized)
        accuracy_validation = accuracy_score(y_validation, y_prediction_validation)
                
        y_prediction_test = model.predict(X_test_normalized)
        accuracy_test = accuracy_score(y_test, y_prediction_test)

        if accuracy_validation > best_accuracy:
            best_model = (model.__class__.__name__, i[1], accuracy_validation, accuracy_test)
            print("New best: ", accuracy_validation, "Test accuracy: ", accuracy_test, i)
            best_accuracy = accuracy_validation

        result = str(model.__class__.__name__) + " " + str(i[1]) + ": " + str(accuracy_validation) + ", " + str(accuracy_test) +"\n"
        print(model.__class__.__name__, i[1], accuracy_validation, accuracy_test)
        test_results.append(result)

    save_test_results(test_results, best_model)
    
    return test_results

def classification_pipeline():
        
    """df = pd.read_csv("tabular_data/listing.csv")
    df = td.clean_tabular_data(df)
    df = df.astype({"guests": "int32", "bedrooms": "int32"})

    X, y = td.load_airbnb(df, "Category")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=47)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5, random_state=37)

    scaler = preprocessing.RobustScaler()
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    X_validation_normalized = scaler.transform(X_validation) """

    X_test_normalised, y_test, X_train_normalised, y_train, X_validation_normalised, y_validation = prepare_data.prepare("Category")
    run_baseline_LogisticRegression_model()
    custom_tune_classification_model_hyperparameters(X_train_normalised, y_train, X_validation_normalised, y_validation, X_test_normalised, y_test)

if __name__ == "__main__":
    classification_pipeline()