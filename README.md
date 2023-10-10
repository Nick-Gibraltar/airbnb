# AI Core Data Science Specialisation Project - Modelling Airbnb's Property Listing Dataset

## Project Description
The Airbnb data modelling project aims to teach data science skills through the analysis of data from Airbnb. To achieve this aim,
multiple models were trained, tuned, and evaluated on practice dataset from Airbnb which was provided for the purpose. The dataset
contained information about approximately 900 listings from Airbnb, with data fields such as price per night, number of bedrooms,
number of bathrooms, guest ratings, etc. being included. The data was analysed using regression, classification, and neural network
models. The primary aim of the project was to use the features contained in the dataset to predict the price per night for Airbnb
accomodation which provided multiple opportunites for learning in so-doing.

The project comprised multiple stages summarised as follows:

1. Set up the development environment.
2. Download the dataset and clean it to, amongst other things, remove rows with missing data, ensure datatypes are consistent,
remove data items that would in other ways lead to errors in processing.
3. Setup, tune, and evaluate a selection of regression models.
4. Setup, tune, and evaluate a selection of classification models.
5. Setup, tune, and evaluate a selection of neural network models.
6. Apply the code written to perform these analyses to a new dataset.
7. Document the project.

## Learning Outcomes
The project gave rise to the following learning outcomes:

- An introduction to a some commonly used models for regression and classification analysis of data.
- Practical experience of the data preparation required to enable the models to be applied to the dataset.
- Practical experience of ways to select the best-performing model through systematic application of a range of models to the dataset
and with a range of parameters being used for each model.
- Understanding of the importance of splitting the data into training, validation, and testing sets.
- Knowledge of some of the important measures of model performance.
- A basic knowledge of the Pytorch machine-learning library as used to build a simple neural network.
- Practical experience of the process by which the performance of multiple neural networks can be analysed in a systematic way

## Regression Analysis
The following regression models were applied to the dataset.
- Stochastic Gradient Descent Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Random Forest Regressor
- Support Vector Regressor

The model types and paramters used are contained in the generate_model_parameters function. A total of 268 distinct model/parameter
configurations were tested. The best result was obtained from the Gradient Boosting Regressor with the following parameters:

- Learning Rate: 0.1
- Maximum Depth: 2
- Number of Estimators: 100

Which gave an r-squared on the test data set of 0.36.

## Classification Analysis
The following classification models were applied to the dataset.
- Stochastic Gradient Descent Classifier
- Gradient Boosting Classifier
- Decision Tree Classifier
- Random Forest Classifier
- Logistic Regression
- Support Vector Classifier

The model types and paramters used are contained in the generate_model_parameters function. A total of 720 distinct model/parameter
configurations were tested. The best result was obtained from the Gradient Boosting Classifier with the following parameters:

- Learning Rate: 0.1
- Number of Estimators: 1000
- Maximum Depth: 3

Which gave an accuracy score of 0.376 on the test data set.

## Neural Network Analysis
A selection of simple neural networks, 20 in total, were also used to analyse the data in an attempt to predict the accommodation
price per night from the features in the dataset. The networks had either one, two, or three hidden layers with the numbers of
neurons in each layer ranging from 4 to 128. The search for the best neural network was necessarily very cursory, the project serving
more as a training ground in the methods that could be used to analyse the data than as a means to identify a definitive "best model",
especially given the small size of the dataset which contains only approximately 900 data items.

The best r-squared was approximately 0.26 which was achieved by 3 of the models as follows (where the numbers indicate the number of
neurons in each layer and with each layer being connected via a ReLU activation function):

11 - 16 - 1 learning rate = 0.01      r-squared = 0.263
11 - 8 - 8 - 1 learning rate = 0.001  r-squared = 0.261
11 - 8 - 8 - 1 learning rate  = 0.01  r-squared = 0.256

with the train:validate:test split being 70:15:15

Given the small size of the dataset, I speculated that the results could be strongly influenced by the choice of the split between train, validate, and test data. Trying an alternative split of 80:10:10 gave a maximum r-squared of 0.395, once again with the 11 - 16 - 1 model but this time with a learning rate of 0.001. No statistical testing was carried out to determine the significance of these results.

![Alt text](../../../../Pictures/Screenshots/Screenshot%20from%202023-10-10%2017-05-32.png)

![Alt text](../../../../Pictures/Screenshots/Screenshot%20from%202023-10-10%2017-08-05.png)

# File Structure
The project code is contained in 4 files: tabular_data.py, regression_modelling.py. classification_modelling.py, and airbnb_neural_network.py. There is also a file containing neural network configurations, nn_config.yaml

The functions contained within tabular_data.py are for the purpose of processing the data prior to the modelling. In each case this module is imported into the modules in which code for data modelling is contained.

# Usage Instructions
In each case the code is simple to use, simply running it with the python3 terminal command on the module containing the chosen modeliing type. Results for each model that is run are output to appropriately named files in sub-directories of the diretory containing the code, as well as the best performing model according to the chosen metric.