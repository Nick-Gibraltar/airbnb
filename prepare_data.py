import pandas as pd
import tabular_data as td
from sklearn import model_selection
from sklearn import preprocessing

def prepare(label, validation=False):

    # Load and clean the Airbnb dataset
    df = pd.read_csv("tabular_data/listing.csv")
    df = td.clean_tabular_data(df)
    df = df.astype({"guests": "int32", "bedrooms": "int32"})
    # Price_Night (for NN and Regression)
    # Category (for classification)

    X, y = td.load_airbnb(df, label)
    category_counts = y.value_counts()
    print(category_counts)
    # Scale the price per night data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=47)
    if validation:
        X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5, random_state=37)

    # Normalise the data
    X_validation_normalised = None
    y_validation = None
    scaler = preprocessing.RobustScaler()
    scaler.fit(X_train)
    X_train_normalised = scaler.transform(X_train)
    X_test_normalised = scaler.transform(X_test)
    if validation:
        X_validation_normalised = scaler.transform(X_validation)    

    return (X_test_normalised, y_test, X_train_normalised, y_train, X_validation_normalised, y_validation)

if __name__ == "__main__":
    prepare("Category")















"""
# Regression
    # Split into train, test, and validation datasets 
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=47)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5, random_state=37)

    # Normalise the data
    scaler = preprocessing.RobustScaler()
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    X_validation_normalized = scaler.transform(X_validation)    


# Classification

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=47)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5, random_state=37)

    scaler = preprocessing.RobustScaler()
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    X_validation_normalized = scaler.transform(X_validation)
    """