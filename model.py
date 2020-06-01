"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
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
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    train_df=feature_vector_df[['Order No', 'User Id', 'Vehicle Type', 'Platform Type',
       'Personal or Business', 'Placement - Day of Month',
       'Placement - Weekday (Mo = 1)', 'Placement - Time',
       'Confirmation - Day of Month', 'Confirmation - Weekday (Mo = 1)',
       'Confirmation - Time', 'Arrival at Pickup - Day of Month',
       'Arrival at Pickup - Weekday (Mo = 1)', 'Arrival at Pickup - Time',
       'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)', 'Pickup - Time',
       'Distance (KM)', 'Temperature',
       'Precipitation in millimeters', 'Pickup Lat', 'Pickup Long',
       'Destination Lat', 'Destination Long', 'Rider Id','No_Of_Orders', 'Age','Average_Rating','No_of_Ratings']]
    

    train_df.columns = [col.replace(' ', '_').lower()
                    for col in train_df.columns]


    # Reorder columns so that the dependent variable is at the end.

    train_df = train_df[['order_no', 'user_id', 'vehicle_type', 'platform_type', 'personal_or_business',
                     'placement_-_day_of_month', 'placement_-_weekday_(mo_=_1)', 'placement_-_time',
                     'confirmation_-_day_of_month', 'confirmation_-_weekday_(mo_=_1)',
                     'confirmation_-_time', 'arrival_at_pickup_-_day_of_month',
                     'arrival_at_pickup_-_weekday_(mo_=_1)', 'arrival_at_pickup_-_time',
                     'pickup_-_day_of_month', 'pickup_-_weekday_(mo_=_1)', 'pickup_-_time',
                     'distance_(km)', 'temperature',
                     'precipitation_in_millimeters', 'pickup_lat', 'pickup_long', 'destination_lat',
                     'destination_long', 'rider_id', 'no_of_orders', 'age', 'average_rating',
                     'no_of_ratings', 'time_from_pickup_to_arrival']]

    # Renaming the columns to make working with the DataFrames easier.

    train_df.columns = ['order_no', 'user_id', 'vehicle_type', 'platform_type', 'personal_or_business',
                    'placement(DOM)', 'placement(weekday)', 'placement(time)', 'confirmation(DOM)',
                    'confirmation(weekday)', 'confirmation(time)', 'arrival_at_pickup(DOM)',
                    'arrival_at_pickup(weekday)', 'arrival_at_pickup(time)', 'pickup(DOM)',
                    'pickup(weekday)', 'pickup(time)',
                    'distance(km)',
                    'temperature', 'precipitation(mm)', 'pickup_lat', 'pickup_long', 'destination_lat',
                    'destination_long', 'rider_id', 'no_of_orders', 'age', 'average_rating',
                    'no_of_ratings', 'time_from_pickup_to_arrival']

    # Using the Pandas .drop() method.
    # Remove columns by specifying the column names and corresponding axis.

    merged_df = train_df.drop('order_no', axis=1)



    # Using the Pandas .replace() method, replace all NaN values with 0.

    merged_df = merged_df.replace(np.nan, 0)


    # Converting time strings to seconds using the Pandas .to_timedelta() method.
    # Using the .dt accessor object for datetimelike properties of the Series values to convert to seconds.

    merged_df['placement(time)'] = pd.to_timedelta(
    merged_df['placement(time)']).dt.total_seconds()

    merged_df['confirmation(time)'] = pd.to_timedelta(
    merged_df['confirmation(time)']).dt.total_seconds()

    merged_df['arrival_at_pickup(time)'] = pd.to_timedelta(
    merged_df['arrival_at_pickup(time)']).dt.total_seconds()

    merged_df['pickup(time)'] = pd.to_timedelta(
    merged_df['pickup(time)']).dt.total_seconds()


    # Using the Pandas .drop() method.
    # Remove columns that are not useful by specifying the column names and corresponding axis.

    merged_df = merged_df.drop(['user_id', 'vehicle_type', 'rider_id'], axis=1)



    # Encoding categorical data using Pandas .get_dummies() method.

    merged_df = pd.get_dummies(merged_df, columns=[
                            'platform_type', 'personal_or_business'], 
                             drop_first=True)

    # Using the Pandas .drop() method.
    # Remove columns that are not useful by specifying the column names and corresponding axis.

    merged_df = merged_df.drop(['precipitation(mm)'], axis=1)

    # Replacing NaN values using the Pandas .fillna() method with the mean of the column.

    merged_df['temperature']= merged_df['temperature'].fillna(merged_df['temperature'].mean())


    # Using the Pandas .drop() method.
    # Remove columns that are not useful by specifying the column names and corresponding axis.

    merged_df = merged_df.drop(['confirmation(DOM)', 'confirmation(weekday)', 'arrival_at_pickup(DOM)', 
                            'arrival_at_pickup(weekday)', 'pickup(DOM)', 'pickup(weekday)'], axis=1)


    # Using the geopy.distance to calculate the distance between the two latitude and longitude points
    from geopy.distance import distance
    def distance_calc(merged_df):
        '''calculate distance (m) between two lat&long points using the Vincenty formula '''

        dist_calc = distance((merged_df.pickup_lat, merged_df.pickup_long),
                         (merged_df.destination_lat, merged_df.destination_long)).km
        return dist_calc


    # Using the lambda function to iterate distance_calc(X) over the X DataFrame and create a new colmun.

    merged_df['distance_(lat/long)_(km)'] = merged_df.apply(lambda r: distance_calc(r), axis=1)



    # Calculating the difference between the lat/long distance and the distance given.

    merged_df['distance_diff']= merged_df['distance(km)']- merged_df['distance_(lat/long)_(km)']

    #Converting the placement day of month into weeks
    merged_df['placement(DOM)'] = round((train_df['placement(DOM)']%365)/7, 0)

    merged_df['placement(DOM)'][merged_df['placement(DOM)'] == 0] = 1

    #Converting placement weekday into weekday or weekend category

    merged_df['placement(weekday)'].mask(merged_df['placement(weekday)'] == 1 , 1, inplace=True)
    merged_df['placement(weekday)'].mask(merged_df['placement(weekday)'] == 2 , 1, inplace=True)
    merged_df['placement(weekday)'].mask(merged_df['placement(weekday)'] == 3 , 1, inplace=True)
    merged_df['placement(weekday)'].mask(merged_df['placement(weekday)'] == 4 , 1, inplace=True)
    merged_df['placement(weekday)'].mask(merged_df['placement(weekday)'] == 5 , 1, inplace=True)
    merged_df['placement(weekday)'].mask(merged_df['placement(weekday)'] == 6 , 0, inplace=True)
    merged_df['placement(weekday)'].mask(merged_df['placement(weekday)'] == 7 , 0, inplace=True)



    # The average time taken from placement time to pickup time in seconds

    merged_df['average_time']= (merged_df['placement(time)'] + merged_df['confirmation(time)'] + 
                            merged_df['arrival_at_pickup(time)'] + merged_df['pickup(time)'])/4


    #Removing Outliers
    size= 21201
    new_df= merged_df[:len(train_df)] 
    y = new_df['time_from_pickup_to_arrival']
    removed_outliers = y.between(y.quantile(.05), y.quantile(.95))

    print(str(y[removed_outliers].size) + "/" + str(size) + " data points remain.") 

    #y[removed_outliers].plot().get_figure()
    index_names = new_df[~removed_outliers].index
    new_df.drop(index_names, inplace=True)

    return new_df
    #predict_vector = feature_vector_df[['Order No', 'User Id', 'Vehicle Type', 'Platform Type',
       #'Personal or Business', 'Placement - Day of Month',
       #'Placement - Weekday (Mo = 1)', 'Placement - Time',
       #'Confirmation - Day of Month', 'Confirmation - Weekday (Mo = 1)',
       #'Confirmation - Time', 'Arrival at Pickup - Day of Month',
       #'Arrival at Pickup - Weekday (Mo = 1)', 'Arrival at Pickup - Time',
       #'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)', 'Pickup - Time',
       #'Distance (KM)', 'Temperature',
       #'Precipitation in millimeters', 'Pickup Lat', 'Pickup Long',
       #'Destination Lat', 'Destination Long', 'Rider Id']]
    
    
    #predict_vector.drop(['Vehicle Type','Order No'], axis=1, inplace=True)


    #predict_vector[['a','b', 'UserId No']] = predict_vector["User Id"].str.split("_", expand = True)
    #predict_vector[['c','d','RiderId No']] = predict_vector["Rider Id"].str.split("_",  expand = True)

    #predict_vector.drop(["User Id",'Rider Id', 'a', 'b','c','d'], axis=1, inplace = True)

    #predict_vector.rename(columns={'UserId No':'User Id'}, inplace=True)
    #predict_vector.rename(columns={'RiderId No':'Rider Id'}, inplace=True)

    #def time_converter(predict_vector):
        #for x in predict_vector.columns:
            #if x.endswith("Time"):
                #predict_vector[x] = pd.to_datetime(predict_vector[x], format='%I:%M:%S %p').dt.strftime("%H:%M:%S")
        #return predict_vector

    #predict_vector = time_converter(predict_vector)
    #predict_vector[['Placement - Time', 'Confirmation - Time' , 'Arrival at Pickup - Time', 'Pickup - Time']][3:6]


    #predict_vector['Placement - Time_Hour'] = pd.to_datetime(predict_vector['Placement - Time']).dt.hour
    #predict_vector['Placement - Time_Minute'] = pd.to_datetime(predict_vector['Placement - Time']).dt.minute
    #predict_vector['Placement - Time_Seconds'] = pd.to_datetime(predict_vector['Placement - Time']).dt.second

    #predict_vector['Confirmation - Time_Hour'] = pd.to_datetime(predict_vector['Confirmation - Time']).dt.hour
    #predict_vector['Confirmation - Time_Minute'] = pd.to_datetime(predict_vector['Confirmation - Time']).dt.minute
    #predict_vector['Confirmation - Time_Seconds'] = pd.to_datetime(predict_vector['Confirmation - Time']).dt.second

    #predict_vector['Arrival at Pickup - Time_Hour'] = pd.to_datetime(predict_vector['Arrival at Pickup - Time']).dt.hour
    #predict_vector['Arrival at Pickup - Time_Minute'] = pd.to_datetime(predict_vector['Arrival at Pickup - Time']).dt.minute
    #predict_vector['Arrival at Pickup - Time_Seconds'] = pd.to_datetime(predict_vector['Arrival at Pickup - Time']).dt.second

    #predict_vector['Pickup - Time_Hour'] = pd.to_datetime(predict_vector['Pickup - Time']).dt.hour
    #predict_vector['Pickup - Time_Minute'] = pd.to_datetime(predict_vector['Pickup - Time']).dt.minute
    #predict_vector['Pickup - Time_Seconds'] = pd.to_datetime(predict_vector['Pickup - Time']).dt.second


    #predict_vector.drop(['Pickup - Time','Arrival at Pickup - Time','Confirmation - Time','Placement - Time'], axis=1, inplace=True)

    #cols = list(copy.columns.values)
    #cols.pop(cols.index('Time from Pickup to Arrival')) 

    #copy = copy[cols+['Time from Pickup to Arrival']]


    #predict_vector['Personal or Business'].unique()
    #Bdict = {'Personal': 0, 'Business': 1}
    #predict_vector['Personal or Business'] = predict_vector['Personal or Business'].map(Bdict)


    #predict_vector= predict_vector.replace(np.nan, 0)

    #copy = copy.drop(['Time from Pickup to Arrival'], axis=1)
    #y = copy['Time from Pickup to Arrival']
    # ------------------------------------------------------------------------



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

def make_prediction(data, model):
    """Prepare request data for model prediciton.

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
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
