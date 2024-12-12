"""Python script designed to facilitate feature selection or removal in a straightforward manner.

In this script, we define a feature_selection function that allows us to include or exclude features 
from the dataset in a SQL-like fashion. This function streamlines the process of feature selection 
and model testing, making it easier to experiment with different subsets of features.
"""


def feature_selection(dataset, test_set=False):
    """Filter the input dataset to include only the selected features.

    Parameters
    ----------
    dataset : pd.dataframe
        The input dataframe with all available features.

    test_set : boolean, optional
        Specifies whether the input dataset is the test one.
    If True, it ensures that target columns ('bike_count' and
    'log_bike_count') are excluded to avoid key errors.
    False by default.

    Returns
    -------
    dataset : pd.dataframe
        The dataset reduced ton the chosen features.
    """
    selected_columns = [
        "counter_id",
        ###########"site_name",
        "date",
        ###########"counter_installation_date",  --> low correlation (<0.01) with target
        ###########"latitude_counter",  --> high correlation (>0.97) with counter_id
        "duree_precip",
        ###########"vent_moyen_10m",  --> high correlation (>0.97) with vent_inst_max
        ###########"vent_max",  --> high correlation (>0.97) with vent_inst_max
        "vent_inst_max",
        ###########"vent_max_3s",  --> high correlation (>0.97) with vent_inst_max
        ###########"point_rosée",
        ###########"temp_min_10cm",  --> high correlation (>0.97) with temp_surface
        ###########"temp_min_50cm",  --> high correlation (>0.97) with temp_surface
        "temp_surface",
        ###########"humidite",  --> reduce the model's performance
        ###########"humidite_min",  --> high correlation (>0.97) with humidite
        ###########"humidite_max",  --> high correlation (>0.97) with humidite
        ###########"duree_humidite_40",  --> high correlation (>0.97) with duree_humidite_80
        "duree_humidite_80",
        ###########"pression_station",   --> low correlation (<0.01) with target
        ###########"visibilite",
        ###########"code_meteo",
        "duree_ensoleillement_utc",
        "precip_1h",
        ###########"temperature",  --> high correlation (>0.97) with temp_surface
        ###########"temp_min",  --> high correlation (>0.97) with temp_surface
        ###########"temp_max",  --> high correlation (>0.97) with temp_surface
        ###########"duree_gel"
        # ------------------------#
        ###########"counter_technical_id",  --> redundant with counter_id
        ###########"counter_name",  --> redundant with counter_id
        ###########"site_id",  --> redundant with site_name
        ###########"coordinates",  --> redundant with latitude_counter and longitude_counter
        ###########"id_poste",  --> redundant with nom_poste
        ###########"pression_mer",  --> high correlation (>0.97) with pression_station
        ###########"nom_poste",  --> constant
        ###########"latitude_poste",  --> constant
        ###########"longitude_poste",  --> constant
        ###########"altitude",  --> constant
        ###########"longitude_counter",  --> low correlation (<0.01) with target
        ###########"direction_vent_10m",  --> low correlation (<0.01) with target
        ###########"direction_vent_max",  --> low correlation (<0.01) with target
        ###########"heure_vent_max",  --> low correlation (<0.01) with target
        ###########"direction_vent_inst_max",  --> low correlation (<0.01) with target
        ###########"heure_vent_inst_max",  --> low correlation (<0.01) with target
        ###########"heure_vent_max_3s",  --> low correlation (<0.01) with target
        ###########"heure_humidite_min",  --> low correlation (<0.01) with target
        ###########"heure_humidite_max",  --> low correlation (<0.01) with target
        ###########"heure_temp_min",  --> low correlation (<0.01) with target
        ###########"heure_temp_max",  --> low correlation (<0.01) with target
    ]

    if not test_set:
        selected_columns = selected_columns + [
            "log_bike_count"
            ###########"bike_count"  --> redundant with log_bike_count
        ]

    return dataset[selected_columns]
