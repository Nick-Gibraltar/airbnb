import tabular_data as td
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def main():
        
    df = pd.read_csv("tabular_data/listing.csv")
    df = td.clean_tabular_data(df)
    df = df.astype({"guests": "int32", "bedrooms": "int32"})

    X, y = td.load_airbnb(df, "Price_Night")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
    #X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)

    normalizer = preprocessing.MinMaxScaler().fit(X)
    X_train_normalized = normalizer.transform(X_train)
    X_test_normalized = normalizer.transform(X_test)
    #X_validation_normalized = normalizer.transform(X_validation)

    SGD_model = SGDRegressor()
    SGD_model.fit(X_train_normalized, y_train)

    plt.scatter(y_test,SGD_model.predict(X_test_normalized))
    plt.grid()
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    plt.title('scatter plot between actual y and predicted y')
    plt.show()
    print("MSE: ", mean_squared_error(y_test,SGD_model.predict(X_test_normalized)))
    

if __name__ == "__main__":
    main()