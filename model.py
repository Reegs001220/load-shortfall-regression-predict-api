"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
import sklearn

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    test_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------

    median_valencia_pressure = test_df['Valencia_pressure'].median()

    test_df['Valencia_pressure'].fillna(median_valencia_pressure, inplace=True)

    test_df['time'] = pd.to_datetime(test_df['time'])

    test_df['time_unix'] = (test_df['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    test_df['Valencia_wind_deg'] = test_df['Valencia_wind_deg'].str.extract('(\d+)', expand=False).astype(float)

    test_df['Seville_pressure'] = test_df['Seville_pressure'].str.extract('(\d+)', expand=False).astype(float)

    test_df['Madrid_temp_range'] = test_df['Madrid_temp_max'] - test_df['Madrid_temp_min']
    test_df['Valencia_temp_range'] = test_df['Valencia_temp_max'] - test_df['Valencia_temp_min']
    test_df['Seville_temp_range'] = test_df['Seville_temp_max'] - test_df['Seville_temp_min']
    test_df['Bilbao_temp_range'] = test_df['Bilbao_temp_max'] - test_df['Bilbao_temp_min']
    test_df['Barcelona_temp_range'] = test_df['Barcelona_temp_max'] - test_df['Barcelona_temp_min']

    test_df['year'] = test_df['time'].dt.year
    test_df['month'] = test_df['time'].dt.month
    test_df['day'] = test_df['time'].dt.day
    test_df['hour'] = test_df['time'].dt.hour

    test_df = test_df.drop('Unnamed: 0', axis=1)

    features_to_normalize = ['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h',
       'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
       'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
       'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
       'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h',
       'Seville_pressure', 'Seville_rain_1h', 'Bilbao_snow_3h',
       'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
       'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id',
       'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id',
       'Valencia_pressure', 'Seville_temp_max', 'Madrid_pressure',
       'Valencia_temp_max', 'Valencia_temp', 'Bilbao_weather_id',
       'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min',
       'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp',
       'Bilbao_temp_min', 'Bilbao_temp', 'Barcelona_temp_min',
       'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp', 'Madrid_temp_min',
       'time_unix', 'Madrid_temp_range',
       'Valencia_temp_range', 'Seville_temp_range', 'Bilbao_temp_range',
       'Barcelona_temp_range', 'year', 'month', 'day', 'hour'
       ]
    
    scaler = sklearn.preprocessing.StandardScaler()

    test_df[features_to_normalize] = scaler.fit_transform(test_df[features_to_normalize])

    predict_vector = test_df.drop(columns=['time'])

    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prep_data.fillna(0.0, inplace=True)

    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction.tolist()
