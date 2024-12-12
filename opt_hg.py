"""Python script designed to run optuna studies.
Made to do hyperparameter tuning of HistGradientBoostingRegressor.

"""

import optuna
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from load_data import load_data
from feature_selector import feature_selection
from null_manager import null_imputer
from feature_engineering import feature_transformer
from preprocessor import preprocessor_generator

# Applying a time series cross validation split

tscv = TimeSeriesSplit(n_splits=5)

# Running all the scripts

train, test = load_data()
train = null_imputer(train)
train = feature_selection(train)
train = feature_transformer(train)
x_train = train.drop(columns=["log_bike_count"])
y_train = train["log_bike_count"]  # defining target
preprocessor = preprocessor_generator(x_train)

# Defining the objective function for the optuna study


def objective(trial):

    # Setting the hyperparameter ranges to explore

    max_iter = trial.suggest_int("max_iter", 1100, 1900)
    max_depth = trial.suggest_int("max_depth", 10, 17)
    learning_rate = trial.suggest_loguniform("learning_rate", 0.02, 0.12)

    # Defining our regressor and our pipeline

    regressor = HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=8,  # fixing a random state to avoid random variations
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "to_dense",
                FunctionTransformer(
                    lambda x: x.toarray() if hasattr(x, "toarray") else x
                ),
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

    return np.mean(rmse_scores)


# Creating optuna study that will try to minimize RMSE with 20 trials

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Displaying the best parameters found

print(f"Best hyperparameters : {study.best_params}")
print(f"Best RMSE : {study.best_value}")


# best parameters found     max_iter=1170, max_depth=12, learning_rate=0.11958816320752756
