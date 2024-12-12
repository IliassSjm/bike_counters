"""Main script to be run on Kaggle.
    
This script goes through all the other scripts to prepare data, then
fit the model, do predictions on test set and write a CSV file for 
submission.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from load_data import load_data
from feature_selector import feature_selection
from feature_engineering import feature_transformer
from preprocessor import preprocessor_generator
from null_manager import null_imputer

# Running all the scripts to prepare data

train, test = load_data(kaggle=True)
train = null_imputer(train)
test = null_imputer(test)
x_test = feature_selection(test, test_set=True)
train = feature_selection(train)
train = feature_transformer(train)
x_test = feature_transformer(x_test)
x_train = train.drop(columns=["log_bike_count"])
y_train = train["log_bike_count"]
preprocessor = preprocessor_generator(x_train)

# Defining our regressor and our pipeline

regressor = HistGradientBoostingRegressor(
    max_iter=1855,
    max_depth=14,
    learning_rate=0.07364924738942269,
    random_state=8,  # fixing a random state to avoid random variations
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "to_dense",
            FunctionTransformer(lambda x: x.toarray() if hasattr(x, "toarray") else x),
        ),
        ("regressor", regressor),
    ]
)

# Fitting the pipeline

pipeline.fit(x_train, y_train)

# Predicting target values for test dataset

test_pred = pipeline.predict(x_test)

# Writing predictions in the appropriate CSV file for submission

results = pd.DataFrame(
    dict(
        Id=np.arange(test_pred.shape[0]),
        log_bike_count=test_pred,
    )
)
results.to_csv("submission.csv", index=False)
print("Done")
