import itertools
import tabular_data as td
import pandas as pd
import numpy as np

from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

def main():
        
    df = pd.read_csv("tabular_data/listing.csv")
    df = td.clean_tabular_data(df)
    df = df.astype({"guests": "int32", "bedrooms": "int32"})

    X, y = td.load_airbnb(df, "Price_Night")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)

    normalizer = preprocessing.MinMaxScaler().fit(X_test)
    X_train_normalized = normalizer.transform(X_train)
    X_test_normalized = normalizer.transform(X_test)
    X_validation_normalized = normalizer.transform(X_validation)

    SGD_model = SGDRegressor()
    SGD_model.fit(X_train_normalized, y_train)

    plt.scatter(y_test,SGD_model.predict(X_test_normalized))
    plt.grid()
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    plt.title('scatter plot between actual y and predicted y')
    plt.show()
    print("Root Mean Squared Error Validation: ", np.sqrt(mean_squared_error(y_validation,SGD_model.predict(X_validation_normalized))))
    print("Root Mean Squared Error Test: ", np.sqrt(mean_squared_error(y_test,SGD_model.predict(X_test_normalized))))
    print("R squared Validation: ", r2_score(y_validation,SGD_model.predict(X_validation_normalized)))
    print("R squared Test: ", r2_score(y_test,SGD_model.predict(X_test_normalized)))

    custom_tune_regression_model_hyperparameters(X_train_normalized, y_train, X_validation_normalized, y_validation)


def custom_tune_regression_model_hyperparameters(X_train_normalized, y_train, X_validation_normalized, y_validation):
    
    model_specifications_list = generate_model_parameters()
    min_sqrt_mse = np.inf

    for i in model_specifications_list:
        print("Running ", i)
        
        model_type = i[0]
        model_parameters = i[1]
    
        model = model_type(**model_parameters)

        model.fit(X_train_normalized, y_train)
        
        sqrt_mse = np.sqrt(mean_squared_error(y_validation, model.predict(X_validation_normalized)))
        if sqrt_mse < min_sqrt_mse:
            r2 = r2_score(y_validation, model.predict(X_validation_normalized))
            print("New best achieved: ", sqrt_mse, )
            best_model = [i, sqrt_mse, r2]
            min_sqrt_mse = sqrt_mse

    print(best_model)



def generate_model_parameters():

    model_specifications  = [
                            [GradientBoostingRegressor, 
                            ["learning_rate", "max_depth", "n_estimators"], 
                            [[0.01, 0.1, 1], [2, 3, 10], [10, 100, 1000]]],

                            [SVR,
                            ["kernel", "C"],
                            [["linear", "rbf", "sigmoid"], [0.1, 1, 10]]],

                            [SGDRegressor,
                            ["penalty", "alpha", "max_iter", "eta0"],
                            [["l1", "l2", "elasticnet"], [0.01, 0.001, 0.0001, 0.00001], [100, 500, 1000, 10000], [0.001, 0.01, 0.1]]],

                            [DecisionTreeRegressor,
                            ["max_depth", "splitter"],
                            [[None, 2, 3, 5], ["best"]]],

                            [RandomForestRegressor,
                            ["n_estimators", "max_depth"],
                            [[10, 100, 1000, 5000], [None, 2, 10]]
                            ]
                            ]    

    model_specifications_list = []
    for i in model_specifications:
        parameter_values = list(itertools.product(*i[2]))
        parameter_keys = i[1]
        for j in parameter_values:
            model_specifications_list.append((i[0], dict(zip(parameter_keys, j))))
    
    return model_specifications_list

if __name__ == "__main__":
    main()