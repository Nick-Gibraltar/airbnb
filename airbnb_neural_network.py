import datetime
import itertools
import json
import tabular_data as td
import pandas as pd
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import yaml
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from sklearn import model_selection

from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error, r2_score


from torch.utils.data import Dataset, DataLoader

def make_data_loader():
    
    df = pd.read_csv("tabular_data/listing.csv")
    df = td.clean_tabular_data(df)
    df = df.astype({"guests": "int32", "bedrooms": "int32"})
    X, y = td.load_airbnb(df, "Price_Night")
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.values.reshape(-1, 1))
    y = pd.DataFrame(y)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=47)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5, random_state=37)

    AirbnbRegressionDataset_Train = AirbnbDataset(X_train, y_train)

    return DataLoader(AirbnbRegressionDataset_Train, batch_size=16, shuffle=True), \
        torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32), \
            torch.tensor(X_validation.values, dtype=torch.float32), torch.tensor(y_validation.values, dtype=torch.float32), \
            torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32)

class AirbnbDataset(Dataset):
    
    def __init__(self, features, labels):
        
        super().__init__()
        self.features = features
        self.labels = labels

    def __getitem__(self, index):

        return torch.tensor(self.features.iloc[index], dtype=torch.float32), torch.tensor(self.labels.iloc[index], dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)

class NN(nn.Module):

    def __init__(self, architecture):
        
        super().__init__()
        self.layers = nn.Sequential(*(architecture[layer]['type'](**architecture[layer]['config']) for layer in architecture))

    def forward(self, X):
                
        return self.layers(X)
    
def train_model(train_loader, X_test, y_test, X_validation, y_validation, X_train, y_train, optimiser_spec, network_spec):
    
    output_dir = os.path.expanduser('~/Documents/AICore/Specialisation/Airbnb_Project/runs')
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_filename = f'model_specifications_{current_datetime}.json'
    output_path = os.path.join(output_dir, output_filename)

    # Save layer specifications to a JSON file
    #with open(output_path, 'w') as f:
    #    json.dump(architecture, f, indent=4)


    model = NN(network_spec)
    optimiser_type, learning_rate = optimiser_spec
    if optimiser_type=="SGD":
        optimiser = optim.SGD(model.parameters(), lr=learning_rate)   
    
    writer = SummaryWriter()

    epochs = 50

    batch_idx = 0
    print(model, learning_rate)
    timer = time.time()
    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            writer.add_scalar("Loss", loss.item(), batch_idx)
            batch_idx += 1
    
    training_duration = time.time() - timer
    prediction_validation = model(X_validation)
    prediction_test = model(X_test)
    timer = time.time()
    prediction_train = model(X_train)
    inference_latency = time.time() - timer
    r2_validation = r_squared(y_validation, prediction_validation)
    r2_test = r_squared(y_test, prediction_test)
    r2_train = r_squared(y_train, prediction_train)

    print(training_duration, inference_latency, r2_validation, r2_test, r2_train)

    #save_model_specification(architecture)

def generate_nn_configs():
    
    with open("nn_config.yaml", "r") as file:
        data = yaml.safe_load(file)

    model_specifications = {}
    for network in data['networks']:
        model_specifications[network['name']] = {
            'optimiser': network['optimiser'],
            'learning_rate': network['learning_rate'],
            'num_hidden_layers': network['num_hidden_layers'],
            'hidden_layer_neurons': network['hidden_layer_neurons']}
    
    configs = []
    for _, model_specs in model_specifications.items():
        architecture = {}
        num_hidden_layers = model_specs['num_hidden_layers']
        for i in range(num_hidden_layers):
            if i==0:
                architecture["linear"+str(i)] = {'type': nn.Linear, 'config': {'in_features': 11, 'out_features': model_specs['hidden_layer_neurons'][i]}}
            else:
                architecture["linear"+str(i)] = {'type': nn.Linear, 'config': {'in_features': model_specs['hidden_layer_neurons'][i-1], 'out_features': model_specs['hidden_layer_neurons'][i]}}
            architecture["relu"+str(i)] = {'type': nn.ReLU, 'config': {}}
        architecture["linear"+str(num_hidden_layers)] = {'type': nn.Linear, 'config': {'in_features': model_specs['hidden_layer_neurons'][num_hidden_layers-1], 'out_features': 1}}
        configs.append(((model_specs["optimiser"], model_specs["learning_rate"]), architecture))

    return configs

def save_model_specification(architecture):

    output_dir = os.path.expanduser('~/Documents/AICore/Specialisation/Airbnb_Project/runs')
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_filename = f'model_specifications_{current_datetime}.json'
    output_path = os.path.join(output_dir, output_filename)

    # Save layer specifications to a JSON file
    with open(output_path, 'w') as f:
        json.dump(architecture, f, indent=4)

def r_squared(y_true, y_pred):
    # Calculate the mean of the true target values
    mean_y_true = torch.mean(y_true)
    
    # Calculate the total sum of squares (TSS)
    tss = torch.sum((y_true - mean_y_true) ** 2)
    
    # Calculate the residual sum of squares (RSS)
    rss = torch.sum((y_true - y_pred) ** 2)
    
    # Calculate R-squared
    r2 = 1 - (rss / tss)
    
    return r2.item()

def main():

    train_loader, X_test, y_test, X_validation, y_validation, X_train, y_train = make_data_loader()
    

    configs = generate_nn_configs()

    for config in configs:
        train_model(train_loader, X_test, y_test, X_validation, y_validation, X_train, y_train, *config)


if __name__ == "__main__":
    main()