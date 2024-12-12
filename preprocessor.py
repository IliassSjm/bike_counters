"""Python script build to ultimately generate a preprocessor for a given dataset.

It is designed to simplify the feature preprocessing "process" in our pipeline.

This script includes functions that:
1. Classify features into categorical, numerical, and date types.
2. Encode date features into multiple components such as year, month, etc.
3. Generate a preprocessor that can be used to transform the dataset.

"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    FunctionTransformer,
    OrdinalEncoder,
    OneHotEncoder,
)
import holidays


def get_features_type(dataset):
    """Classify the features of the dataset into categorical, numerical, and date features.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input dataframe.

    Returns
    -------
    categorical_features : list
        A list of column names corresponding to categorical features.
    Corresponding to 'category' and 'object' dtypes.

    numerical_features : list
        A list of column names corresponding to numerical features.
    Corresponding to 'float' and 'int' dtypes.

    date_features : list
        A list of column names corresponding to date features.
    Corresponding to 'datetime' dtype.
    """
    categorical_features = [
        col for col in dataset.columns if dataset[col].dtype in ["category", "object"]
    ]
    numerical_features = [
        col
        for col in dataset.columns
        if dataset[col].dtype in ["float64", "int64", "int32"]
    ]
    date_features = [
        col for col in dataset.columns if dataset[col].dtype == "datetime64[us]"
    ]

    return categorical_features, numerical_features, date_features


def date_encoder(dataset):
    """Encode date features into multiple components.
    
    - year, month, day, hour, weekday
    - binary features indicating whether the date is a weekend or a holiday
    - binary features indicating whether the date corresponds to peak hours during the working days or weekend.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input dataframe containing the date features to be encoded.

    Returns
    -------
    dataset : pd.DataFrame
        The dataset with the additional date-related features and the original date features removed.
    """
    fr_holidays = holidays.France(years=dataset["date"].dt.year.unique().tolist())

    for col in dataset.columns:
        dataset[f"{col}_year"] = dataset[col].dt.year
        dataset[f"{col}_month"] = dataset[col].dt.month
        dataset[f"{col}_day"] = dataset[col].dt.day
        dataset[f"{col}_hour"] = dataset[col].dt.hour
        dataset[f"{col}_weekday"] = dataset[col].dt.weekday
        dataset[f"{col}_is_weekend"] = (
            dataset[f"{col}_weekday"].isin([5, 6]).astype(int)
        )
        dataset[f"{col}_is_holiday"] = (
            dataset[col].dt.date.isin(fr_holidays).astype(int)
        )
        dataset["is_night"] = (
            (dataset[col].dt.hour >= 23) & (dataset[col].dt.hour <= 4)
        ).astype(int)
        dataset["is_morning_peak_hours_working_day"] = (
            (dataset[col].dt.hour >= 6)
            & (dataset[col].dt.hour <= 7)
            & (dataset[f"{col}_is_weekend"] == 0)
        ).astype(int)
        dataset["is_afternoon_peak_hours_working_day"] = (
            (dataset[col].dt.hour >= 15)
            & (dataset[col].dt.hour <= 18)
            & (dataset[f"{col}_is_weekend"] == 0)
        ).astype(int)
        dataset["is_afternoon_peak_hours_week_end"] = (
            (dataset[col].dt.hour >= 13)
            & (dataset[col].dt.hour <= 16)
            & (dataset[f"{col}_is_weekend"] == 1)
        ).astype(int)

        dataset.drop(columns=[col], inplace=True)

    return dataset


def preprocessor_generator(dataset):
    """Generate a preprocessor for the input dataset that applies various transformations.

    - OneHotEncoder for categorical features
    - StandardScaler for numerical features
    - Date encoding for date features (using the date_encoder function)

    Parameters
    ----------
    dataset : pd.DataFrame
        The input dataframe containing the features to be preprocessed.

    Returns
    -------
    preprocessor : sklearn.compose._column_transformer.ColumnTransformer
        A ColumnTransformer to use in our pipeline.
    """
    categorical_features, numerical_features, date_features = get_features_type(dataset)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat_to_OHE",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("num", StandardScaler(), numerical_features),
            ("date", FunctionTransformer(date_encoder, validate=False), date_features),
        ]
    )
    return preprocessor
