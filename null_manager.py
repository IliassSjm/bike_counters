"""Python script designed to perform operations related to missing data management.

In this script, we define a null_imputer function that applies various methods to manage 
missing data in both train and test datasets, such as imputing values with the mean or 
median, or removing rows based on prior analysis and experimentation.
"""


def null_imputer(dataset):
    """Manage missing values in the input dataset by applying imputation methods.

    Parameters
    ----------
    dataset : pd.dataframe
        The input dataframe '(either train or test dataset) with missing values.
    null values.

    Returns
    -------
    dataset : pd.dataframe
        The dataset appropriately handled.
    """
    # Imputing missing values of several attributes

    dataset["temp_surface"] = dataset["temp_surface"].fillna(
        dataset["temp_surface"].mean()
    )
    dataset["precip_1h"] = dataset["precip_1h"].fillna(dataset["precip_1h"].median())
    dataset["vent_inst_max"] = dataset["vent_inst_max"].fillna(
        dataset["vent_inst_max"].median()
    )
    dataset["duree_precip"] = dataset["duree_precip"].fillna(
        dataset["duree_precip"].median()
    )

    dataset["temp_min"] = dataset["temp_min"].fillna(dataset["temp_min"].mean())
    dataset["heure_temp_min"] = dataset["heure_temp_min"].fillna(
        dataset["heure_temp_min"].mean()
    )
    dataset["temp_max"] = dataset["temp_max"].fillna(dataset["temp_max"].mean())
    dataset["heure_temp_max"] = dataset["heure_temp_max"].fillna(
        dataset["heure_temp_max"].mean()
    )
    dataset["duree_gel"] = dataset["duree_gel"].fillna(dataset["duree_gel"].median())

    return dataset
