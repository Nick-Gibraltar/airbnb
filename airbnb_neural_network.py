import datetime
import json
import os
import pandas as pd
import tabular_data as td
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import uuid
import yaml

from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

def make_data_loader():
    
    # Load and clean the Airbnb dataset
    df = pd.read_csv("tabular_data/listing.csv")
    df = td.clean_tabular_data(df)
    df = df.astype({"guests": "int32", "bedrooms": "int32"})
    X, y = td.load_airbnb(df, "Price_Night")
    
    # Scale the price per night data
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.values.reshape(-1, 1))
    y = pd.DataFrame(y)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=47)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5, random_state=37)

    # The training dataset is an instance of the AirbnbDataset class
    AirbnbRegressionDataset_Train = AirbnbDataset(X_train, y_train)

    # Return the training dataset as a dataloader for easy batching plus the test, train, and validation sets
    return DataLoader(AirbnbRegressionDataset_Train, batch_size=16, shuffle=True), \
            torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32), \
            torch.tensor(X_validation.values, dtype=torch.float32), torch.tensor(y_validation.values, dtype=torch.float32), \
            torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32)

class AirbnbDataset(Dataset):
    # Standard definition for a dataset with the required __getitem__ and __len__ methods defined
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
        # Use dictionary unpacking and list comprehension to create the network 
        self.layers = nn.Sequential(*(architecture[layer]['type'](**architecture[layer]['config']) for layer in architecture))

    def forward(self, X):
                
        return self.layers(X)

def serialize_nn_sequential(model):

    # Function to allow generation of the JSON file containing the network configuration by converting
    # non-serializable classes into strings bearing the name of the class
    config = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            layer_info = {
                'type': str(layer.__class__.__name__),
                'config': {'in_features': layer.in_features, 'out_features': layer.out_features}
            }
        else:
            layer_info = {
                'type': str(layer.__class__.__name__),
            }
        config.append(layer_info)
    return config

def train_model(train_loader, X_test, y_test, X_validation, y_validation, X_train, y_train, optimiser_spec, network_spec):
    
    # Generate path names for metric and configuration outputs 
    output_dir = os.path.expanduser('~/Documents/AICore/Specialisation/Airbnb_Project/runs')
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    serial_number = str(uuid.uuid4())
    
    model = NN(network_spec)

    #Save model specifications to a JSON file
    output_filename = f'model_specifications_{current_datetime}.json'
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(serial_number, f, indent=4)
        json.dump(serialize_nn_sequential(model.layers), f, indent=4)
        json.dump(optimiser_spec, f, indent=4)

    # The optimiser type and learning rate are contained in the optimiser_spec tuple 
    optimiser_type, learning_rate = optimiser_spec

    # TODO: if an optimiser other than SGD is required, then this needs to be implemented
    if optimiser_type=="SGD":
        optimiser = optim.SGD(model.parameters(), lr=learning_rate)   
    
    writer = SummaryWriter()
    epochs = 10
    batch_idx = 0
    print(model, learning_rate)
    training_duration_timer = time.time()
    inference_latency_total = 0
    
    # Standard pytorch training loop
    for _ in range(epochs):
        for batch in train_loader:
            features, labels = batch
            inference_latency_timer = time.time()
            prediction = model(features)
            inference_latency_total += time.time() - inference_latency_timer
            loss = F.mse_loss(prediction, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            writer.add_scalar("Loss", loss.item(), batch_idx)
            batch_idx += 1
    
    training_duration = time.time() - training_duration_timer
    inference_latency = inference_latency_total / batch_idx
    
    prediction_validation = model(X_validation)
    prediction_test = model(X_test)
    prediction_train = model(X_train)

    
    r2_train = r_squared(y_train, prediction_train)
    r2_validation = r_squared(y_validation, prediction_validation)
    r2_test = r_squared(y_test, prediction_test)

    mse_train = F.mse_loss(prediction_train, y_train).item()
    mse_validation = F.mse_loss(prediction_validation, y_validation).item()
    mse_test = F.mse_loss(prediction_test, y_test).item()

    metrics_dictionary = {"model_id": current_datetime + "_" + serial_number,
                         "r2_train": r2_train,
                         "r2_validation": r2_validation,
                         "r2_test": r2_test,
                         "mse_train": mse_train,
                         "mse_validation": mse_validation,
                         "mse_test": mse_test,
                         "training_duration": training_duration,
                         "inference_latency": inference_latency}

    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, 'a') as f:
        json.dump(metrics_dictionary, f, indent=4)

    output_path = os.path.join(output_dir, "trained_model_" + current_datetime + "_" + serial_number + ".pt")
    torch.save(model.state_dict(), output_path)

def generate_nn_configs():
    
    # Open yaml file containing network configurations
    with open("nn_config.yaml", "r") as file:
        data = yaml.safe_load(file)

    # Parse into a dictionary
    model_specifications = {}
    for network in data['networks']:
        model_specifications[network['name']] = {
            'optimiser': network['optimiser'],
            'learning_rate': network['learning_rate'],
            'num_hidden_layers': network['num_hidden_layers'],
            'hidden_layer_neurons': network['hidden_layer_neurons']}
    
    # Generate configs list from the dictionary to contain the architecture of the network and optimiser type and learning rate
    # This list is unpacked in the __init__ of the NN class
    configs = []
    for _, model_specs in model_specifications.items():
        architecture = {}
        num_hidden_layers = model_specs['num_hidden_layers']
        print(num_hidden_layers)
        for i in range(num_hidden_layers):
            if i==0:
                architecture["linear"+str(i)] = {'type': nn.Linear, 'config': {'in_features': 11, 'out_features': model_specs['hidden_layer_neurons'][i]}}
            else:
                architecture["linear"+str(i)] = {'type': nn.Linear, 'config': {'in_features': model_specs['hidden_layer_neurons'][i-1],
                                                                               'out_features': model_specs['hidden_layer_neurons'][i]}}
            architecture["relu"+str(i)] = {'type': nn.ReLU, 'config': {}}
        architecture["linear"+str(num_hidden_layers)] = {'type': nn.Linear, 'config': {'in_features': model_specs['hidden_layer_neurons'][num_hidden_layers-1], 'out_features': 1}}
        configs.append(((model_specs["optimiser"], model_specs["learning_rate"]), architecture))

    return configs

def r_squared(y_true, y_pred):
    # r_squared = 1 - (residual sum of squares / total sum of squares)

    mean_y_true = torch.mean(y_true)
    tss = torch.sum((y_true - mean_y_true) ** 2)
    rss = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (rss / tss)
    
    return r2.item()

def main():

    train_loader, X_test, y_test, X_validation, y_validation, X_train, y_train = make_data_loader()
    configs = generate_nn_configs()

    for config in configs:
        train_model(train_loader, X_test, y_test, X_validation, y_validation, X_train, y_train, *config)

if __name__ == "__main__":
    main()