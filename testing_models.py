"""Python script designed to test models.

(By changing hyperparameters or the whole model). 
This script goes through all the other script to prepare
data before training the model and testing it with a time series split. 
Score and execution time are printed at the end.

"""

import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from load_data import load_data
from feature_selector import feature_selection
from feature_engineering import feature_transformer
from preprocessor import preprocessor_generator
from null_manager import null_imputer

# Starting the timer

start_time = time.time()

# Applying a time series cross validation split

tscv = TimeSeriesSplit(n_splits=5)

# Running all the scripts to prepare data

train, test = load_data()
train = null_imputer(train)
train = feature_selection(train)
train = feature_transformer(train)
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
        ),  # to avoid "sparse matrix" issues
        ("model", regressor),
    ]
)

# Setting the cross validation system

rmse_scores = []

for train_index, val_index in tscv.split(x_train):
    X_train_fold, X_val_fold = x_train.iloc[train_index], x_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    pipeline.fit(X_train_fold, y_train_fold)

    y_pred = pipeline.predict(X_val_fold)

    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
    rmse_scores.append(rmse)

# Ending timer
end_time = time.time()
running_time = end_time - start_time

# Displaying results

print(f"Average RMSE over 5 timer series fold : {np.mean(rmse_scores):.5f}")
print(
    f"Execution time : {int(running_time / 60)} minutes and {running_time % 60:.2f} seconds."
)
