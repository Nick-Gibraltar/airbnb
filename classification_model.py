import itertools
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

    X, y = td.load_airbnb(df, "Category")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=47)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5, random_state=37)

    scaler = preprocessing.RobustScaler()
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    X_validation_normalized = scaler.transform(X_validation)

    