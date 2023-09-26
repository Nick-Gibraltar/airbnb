#AI Core Data Science Specialisation Project - Modelling Airbnb's Property Listing Dataset

The Airbnb data modelling project aims to teach data science skills through the analysis of data from Airbnb. To achieve this aim,
multiple models were trained, tuned, and evaluated on practice dataset from Airbnb which was provided for the purpose. The dataset
contained information about approximately 900 listings from Airbnb, with data fields such as price per night, number of bedrooms,
number of bathrooms, guest ratings, etc. being included. The data was analysed using regression, classification, and neural network
models.

The project comprised multiple stages summarised as follows:

1. Set up the development environment.
2. Download the dataset and clean it to, amongst other things, remove rows with missing data, ensure datatypes are consistent,
remove data items that would in other ways lead to errors in processing.
3. Setup, tune, and evaluate a selection of regression models.
4. Setup, tune, and evaluate a selection of classification models.
5. Setup, tune, and evaluate a selection of neural network models.
6. Apply the code written to perform these analyses to a new dataset.
7. Document the project.

Milestones 1 and 2 set up the environment for the project and provide a high-level overview of the system to be built.

Milestone 3 requires some simple code to be written to prepare the data for analysis. The code, contained in the module tabular_data.py, loads data previously downloaded from Amazon AWS into a Pandas dataframe.
Some simple processing is executed as follows: (a) remove rows with certain missing data; (b) concatenate descriptive text that has the appearance of a Python list into a string; and (c) fill in certain items of missing data.

Project Title
Table of Contents, if the README file is long
A description of the project: what it does, the aim of the project, and what you learned
Installation instructions
Usage instructions
File structure of the project
License information