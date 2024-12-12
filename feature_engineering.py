"""Python script designed to do feature engineering.

In this script, we define a feature_transformer function which performs
all feature engineering operations (except the ones on datetime features
which are performed in the date encoder).
"""


def feature_transformer(dataset):
    """Apply all feature engineering transformations.

    Parameters
    ----------
    dataset : pd.dataframe
        The input dataframe.

    Returns
    -------
    dataset : pd.dataframe
        The dataset with newly created features.
    """
    # about rain
    dataset["no_rain"] = (dataset["precip_1h"] == 0).astype(int)
    dataset["weak_rain"] = (
        (dataset["precip_1h"] > 0) & (dataset["precip_1h"] < 2)
    ).astype(int)
    dataset["moderate_rain"] = (
        (dataset["precip_1h"] >= 2) & (dataset["precip_1h"] < 7)
    ).astype(int)

    return dataset
