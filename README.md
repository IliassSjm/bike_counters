# Introduction

This repository contains all the scripts we used to submit predictions 
for the MSDB 2024 kaggle challenge on bike demand in Paris. 

# Access the data needed to run the code

In addition to the datasets provided in the Kaggle competition, 
we sourced an external dataset on weather conditions in Paris. 
The main difference between this dataset and the one originally 
provided is that it contains hourly data.

You can download it via the following link : 
https://www.data.gouv.fr/fr/datasets/r/a77b4d44-d361-4e59-b6cc-cbbf435a2d89

Dataset license : 
Licence Ouverte / Open Licence version 2.0
https://www.etalab.gouv.fr/licence-ouverte-open-licence/

# Repository Structure  

The repository contains 8 Python scripts.

Data Preparation Scripts :

**load_data.py**: Loads all the necessary data.
**feature_selector.py**: Includes/excludes features from the dataset.
**null_manager.py**: Handles null values in the dataset.
**feature_engineering.py**: Performs feature transformations.
**preprocessor.py**: Generates a preprocessor tailored to the data.

Hyperparameter Tuning Script:

**opt_hg.py**: Tunes hyperparameters for the ```HistGradientBoostingRegressor```.

Main Scripts : 

**testing_models.py**: Test the current model.
**kaggle_script.py**: Used on Kaggle to generate predictions.

For more details, feel free to consult the documentation inside the respective scripts.



Thank you for reviewing our code!
Iliass & Pierre.
