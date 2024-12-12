"""Python script made to load all the data needed.

In this script, we define a load_data function built to read all the data files
we have (the ones available in the competition dataset containing bike counts and
the weather dataset sourced externally by us with meteorological observations 
-link available in the README file-) and return complete train and test set.

To run this script locally, ensure you have downloaded the required data files. 
Update the paths in the else section of the load_data function to point to your 
local files.
"""

import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def load_data(kaggle=False):
    """Load all data files, merge them appropriately and return the train and test dataframe.

    Parameters
    ----------
    kaggle : boolean, optional
        Whether to use Kaggle paths for accessing the data. If False, local paths
    are used. False by default.

    Returns
    -------
    train, test : tuple of two pd.dataframes
        Train and test dataset available on the competition dataset enriched with the
    weather data (including precipitation, temperature, wind information, etc.). Train
    is sorted by dates.
    """
    # downloading the data

    if kaggle:
        train_path = "/kaggle/input/msdb-2024/train.parquet"
        test_path = "/kaggle/input/msdb-2024/final_test.parquet"
        weather_path = (
            "/kaggle/input/weather-data-self-sourced/H_75_previous-2020-2022.csv"
        )
        weather_data = pd.read_csv(weather_path, sep=";")
    else:
        train_path = Path("data") / "train.parquet"
        test_path = Path("data") / "final_test.parquet"
        weather_path = Path("weather_data") / "H_75_previous-2020-2022.csv.gz"
        weather_data = pd.read_csv(weather_path, compression="gzip", sep=";")

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    # weather data cleaning and formatting

    weather_data = weather_data.dropna(axis=1, how="all")

    weather_filtered = weather_data[
        [
            "NUM_POSTE",
            "NOM_USUEL",
            "LAT",
            "LON",
            "ALTI",
            "AAAAMMJJHH",
            "RR1",
            "DRR1",
            "FF",
            "DD",
            "FXY",
            "DXY",
            "HXY",
            "FXI",
            "DXI",
            "HXI",
            "FXI3S",
            "HFXI3S",
            "T",
            "TD",
            "TN",
            "HTN",
            "TX",
            "HTX",
            "DG",
            "TNSOL",
            "TN50",
            "TCHAUSSEE",
            "U",
            "UN",
            "HUN",
            "UX",
            "HUX",
            "DHUMI40",
            "DHUMI80",
            "PMER",
            "PSTAT",
            "VV",
            "WW",
            "INS",
            "INS2",
        ]
    ]

    weather_filtered = weather_filtered.rename(
        columns={
            "NUM_POSTE": "id_poste",
            "NOM_USUEL": "nom_poste",
            "LAT": "latitude",
            "LON": "longitude",
            "ALTI": "altitude",
            "AAAAMMJJHH": "date",
            "RR1": "precip_1h",
            "DRR1": "duree_precip",
            "FF": "vent_moyen_10m",
            "DD": "direction_vent_10m",
            "FXY": "vent_max",
            "DXY": "direction_vent_max",
            "HXY": "heure_vent_max",
            "FXI": "vent_inst_max",
            "DXI": "direction_vent_inst_max",
            "HXI": "heure_vent_inst_max",
            "FXI3S": "vent_max_3s",
            "HFXI3S": "heure_vent_max_3s",
            "T": "temperature",
            "TD": "point_rosée",
            "TN": "temp_min",
            "HTN": "heure_temp_min",
            "TX": "temp_max",
            "HTX": "heure_temp_max",
            "DG": "duree_gel",
            "TNSOL": "temp_min_10cm",
            "TN50": "temp_min_50cm",
            "TCHAUSSEE": "temp_surface",
            "U": "humidite",
            "UN": "humidite_min",
            "HUN": "heure_humidite_min",
            "UX": "humidite_max",
            "HUX": "heure_humidite_max",
            "DHUMI40": "duree_humidite_40",
            "DHUMI80": "duree_humidite_80",
            "PMER": "pression_mer",
            "PSTAT": "pression_station",
            "VV": "visibilite",
            "WW": "code_meteo",
            "INS": "duree_ensoleillement_utc",
            "INS2": "duree_ensoleillement_tsv",
        }
    )

    weather_filtered["date"] = pd.to_datetime(
        weather_filtered["date"], format="%Y%m%d%H"
    ).astype("datetime64[us]")
    weather_filtered = weather_filtered[
        weather_filtered["id_poste"] != 75114007
    ]  # 75114007 double of 75114001
    weather_filtered = weather_filtered[
        weather_filtered["id_poste"] != 75107005
    ]  # 75107005 too many missing values
    weather_filtered = weather_filtered[
        weather_filtered["id_poste"] != 75116008
    ]  # 75116008 too far away from most counter

    # The weather dataset is split into two parts (based on null analysis):
    # - Attributes that can be taken from the nearest station to each counter.
    # - Attributes that are only available from a single station (null in the others)
    #   and therefore cannot be analyzed on a local scale.

    weather_global = weather_filtered[weather_filtered["id_poste"] == 75114001]
    weather_global = weather_global[
        [
            "nom_poste",
            "latitude",
            "longitude",
            "altitude",
            "date",
            "duree_precip",
            "vent_moyen_10m",
            "direction_vent_10m",
            "vent_max",
            "direction_vent_max",
            "heure_vent_max",
            "vent_inst_max",
            "direction_vent_inst_max",
            "heure_vent_inst_max",
            "vent_max_3s",
            "heure_vent_max_3s",
            "point_rosée",
            "temp_min_10cm",
            "temp_min_50cm",
            "temp_surface",
            "humidite",
            "humidite_min",
            "heure_humidite_min",
            "humidite_max",
            "heure_humidite_max",
            "duree_humidite_40",
            "duree_humidite_80",
            "pression_mer",
            "pression_station",
            "visibilite",
            "code_meteo",
            "duree_ensoleillement_utc",
        ]
    ]

    weather_local = weather_filtered[
        [
            "id_poste",
            "date",
            "precip_1h",
            "temperature",
            "temp_min",
            "heure_temp_min",
            "temp_max",
            "heure_temp_max",
            "duree_gel",
        ]
    ]

    # Using a sklearn K-D-Tree to build a panda dataframe which to each counter
    # associates the corresponding nearest weather station.

    unique_weather_coords = weather_filtered[
        ["id_poste", "latitude", "longitude"]
    ].drop_duplicates()
    unique_train_coords = train[
        ["counter_id", "latitude", "longitude"]
    ].drop_duplicates()

    weather_tree = cKDTree(unique_weather_coords[["latitude", "longitude"]].values)

    _, nearest_indices = weather_tree.query(
        unique_train_coords[["latitude", "longitude"]].values, workers=-1
    )

    nearest_station_by_counter = pd.DataFrame(
        {
            "counter_id": unique_train_coords["counter_id"].values,
            "id_poste": unique_weather_coords.iloc[nearest_indices]["id_poste"].values,
        }
    )

    # merging weather_global and weather_local to the train and test dataframes.

    train = pd.merge(train, nearest_station_by_counter, on="counter_id", how="left")
    train = pd.merge(
        train, weather_global, on="date", how="left", suffixes=["_counter", "_poste"]
    )
    train = pd.merge(train, weather_local, on=["date", "id_poste"], how="left")
    train.sort_values("date", inplace=True)

    test["orig_index"] = np.arange(test.shape[0])  # for safety matter
    test = pd.merge(test, nearest_station_by_counter, on="counter_id", how="left")
    test = pd.merge(
        test, weather_global, on="date", how="left", suffixes=["_counter", "_poste"]
    )
    test = pd.merge(test, weather_local, on=["date", "id_poste"], how="left")

    test = test.sort_values("orig_index")  # for safety matter
    del test["orig_index"]

    return train, test
