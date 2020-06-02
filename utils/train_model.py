"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
#Import python librariesss
import pandas as pd
import pickle
import numpy as np

#Training the XGBoost regression model on the split data
import xgboost as xgb

#Training the simple and multiple linear regression model on the split data
#from sklearn.linear_model import *
#from statsmodels.api import OLS 
#from statsmodels.api import add_constant
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.ensemble import*
#from sklearn.svm.classes import *
#from sklearn import *
#from sklearn.gaussian_process import *
#from sklearn.isotonic import *
#from sklearn.neighbors import *
#from sklearn.neural_network import *
#from sklearn.tree import *

# Using the distance package from geopy to calculate the distance between coordinates
from geopy.distance import distance

# Train_test_split used to split the x dataframe into the training set and test set
from sklearn.model_selection import train_test_split

# Fetch training data and preprocess for modeling
#train = pd.read_csv('data/train_data.csv')
#rider = pd.read_csv('https://raw.githubusercontent.com/thembeks/Regression-Sendy-Logistics-Challenge-Team-14/Predict/Riders.csv')
#train= pd.read_csv('https://raw.githubusercontent.com/thembeks/Regression-Sendy-Logistics-Challenge-Team-14/Predict/Train.csv')
#train = train.merge(riders, how='left', on='Rider Id')

# Importing the Riders.csv file from github as a Pandas DataFrame.

rider_df = pd.read_csv(
    'https://raw.githubusercontent.com/thembeks/Regression-Sendy-Logistics-Challenge-Team-14/Predict/Riders.csv')

# Replacing all the blank spaces between words in the column names with an underscore.

#rider_df.columns = [col.replace(' ', '_').lower()
                    #for col in rider_df.columns]



# Importing the Train.csv file from github as a Pandas DataFrame.

train_df = pd.read_csv(
    'https://raw.githubusercontent.com/thembeks/Regression-Sendy-Logistics-Challenge-Team-14/Predict/Train.csv')

# Replacing all the blank spaces between words in the column names with an underscore.

#train_df.columns = [col.replace(' ', '_').lower()
                    #for col in train_df.columns]



# Merge the train and rider DataFrame using Pandas .merge() function
# Training datasets


train_df = pd.merge(train_df, rider_df, on='Rider Id')

# Reorder columns so that the dependent variable is at the end.

#train_df = train_df[['order_no', 'user_id', 'vehicle_type', 'platform_type', 'personal_or_business',
                     #'placement_-_day_of_month', 'placement_-_weekday_(mo_=_1)', 'placement_-_time',
                     #'confirmation_-_day_of_month', 'confirmation_-_weekday_(mo_=_1)',
                     #'confirmation_-_time', 'arrival_at_pickup_-_day_of_month',
                     #'arrival_at_pickup_-_weekday_(mo_=_1)', 'arrival_at_pickup_-_time',
                     #'pickup_-_day_of_month', 'pickup_-_weekday_(mo_=_1)', 'pickup_-_time',
                     #'arrival_at_destination_-_day_of_month', 'arrival_at_destination_-_weekday_(mo_=_1)',
                     #'arrival_at_destination_-_time', 'distance_(km)', 'temperature',
                     #'precipitation_in_millimeters', 'pickup_lat', 'pickup_long', 'destination_lat',
                     #'destination_long', 'rider_id', 'no_of_orders', 'age', 'average_rating',
                     #'no_of_ratings', 'time_from_pickup_to_arrival']]

# Renaming the columns to make working with the DataFrames easier.

#train_df.columns = ['order_no', 'user_id', 'vehicle_type', 'platform_type', 'personal_or_business',
                    #'placement(DOM)', 'placement(weekday)', 'placement(time)', 'confirmation(DOM)',
                    #'confirmation(weekday)', 'confirmation(time)', 'arrival_at_pickup(DOM)',
                    #'arrival_at_pickup(weekday)', 'arrival_at_pickup(time)', 'pickup(DOM)',
                    #'pickup(weekday)', 'pickup(time)', 'arrival_at_destination(DOM)',
                    #'arrival_at_destination(weekday)', 'arrival_at_destination(time)', 'distance(km)',
                    #'temperature', 'precipitation(mm)', 'pickup_lat', 'pickup_long', 'destination_lat',
                    #'destination_long', 'rider_id', 'no_of_orders', 'age', 'average_rating',
                    #'no_of_ratings', 'time_from_pickup_to_arrival']

# Using the Pandas .drop() method.
# Remove columns by specifying the column names and corresponding axis.

merged_df = train_df.drop(['Arrival at Destination - Day of Month', 'Arrival at Destination - Weekday (Mo = 1)',
                              'Arrival at Destination - Time','Order No'], axis=1)



# Using the Pandas .replace() method, replace all NaN values with 0.

merged_df = merged_df.replace(np.nan, 0)


# Converting time strings to seconds using the Pandas .to_timedelta() method.
# Using the .dt accessor object for datetimelike properties of the Series values to convert to seconds.

merged_df['Placement - Time'] = pd.to_timedelta(
merged_df['Placement - Time']).dt.total_seconds()

merged_df['Confirmation - Time'] = pd.to_timedelta(
merged_df['Confirmation - Time']).dt.total_seconds()

merged_df['Arrival at Pickup - Time'] = pd.to_timedelta(
merged_df['Arrival at Pickup - Time']).dt.total_seconds()

merged_df['Pickup - Time'] = pd.to_timedelta(
merged_df['Pickup - Time']).dt.total_seconds()


# Using the Pandas .drop() method.
# Remove columns that are not useful by specifying the column names and corresponding axis.

merged_df = merged_df.drop(['User Id', 'Vehicle Type', 'Rider Id'], axis=1)



# Encoding categorical data using Pandas .get_dummies() method.

merged_df = pd.get_dummies(merged_df, columns=[
                            'Platform Type', 'Personal or Business'], 
                             drop_first=True)

# Using the Pandas .drop() method.
# Remove columns that are not useful by specifying the column names and corresponding axis.

merged_df = merged_df.drop(['Precipitation in millimeters'], axis=1)

# Replacing NaN values using the Pandas .fillna() method with the mean of the column.

merged_df['Temperature']= merged_df['Temperature'].fillna(merged_df['Temperature'].mean())


# Using the Pandas .drop() method.
# Remove columns that are not useful by specifying the column names and corresponding axis.

merged_df = merged_df.drop(['Confirmation - Day of Month', 'Confirmation - Weekday (Mo = 1)', 'Arrival at Pickup - Day of Month', 
                            'Arrival at Pickup - Weekday (Mo = 1)', 'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)'], axis=1)


# Using the geopy.distance to calculate the distance between the two latitude and longitude points
from geopy.distance import distance
def distance_calc(merged_df):
    '''calculate distance (m) between two lat&long points using the Vincenty formula '''

    dist_calc = distance((merged_df['Pickup Lat'], merged_df['Pickup Long']),
                         (merged_df['Destination Lat'], merged_df['Destination Long'])).km
    return dist_calc


# Using the lambda function to iterate distance_calc(X) over the X DataFrame and create a new colmun.

merged_df['distance_(lat/long)_(km)'] = merged_df.apply(lambda r: distance_calc(r), axis=1)



# Ccalculating the difference between the lat/long distance and the distance given.

merged_df['distance_diff']= merged_df['Distance (KM)']- merged_df['distance_(lat/long)_(km)']

#Converting the placement day of month into weeks
merged_df['Placement - Day of Month'] = round((train_df['Placement - Day of Month']%365)/7, 0)

merged_df['Placement - Day of Month'][merged_df['Placement - Day of Month'] == 0] = 1

#Converting placement weekday into weekday or weekend category

merged_df['Placement - Weekday (Mo = 1)'].mask(merged_df['Placement - Weekday (Mo = 1)'] == 1 , 1, inplace=True)
merged_df['Placement - Weekday (Mo = 1)'].mask(merged_df['Placement - Weekday (Mo = 1)'] == 2 , 1, inplace=True)
merged_df['Placement - Weekday (Mo = 1)'].mask(merged_df['Placement - Weekday (Mo = 1)'] == 3 , 1, inplace=True)
merged_df['Placement - Weekday (Mo = 1)'].mask(merged_df['Placement - Weekday (Mo = 1)'] == 4 , 1, inplace=True)
merged_df['Placement - Weekday (Mo = 1)'].mask(merged_df['Placement - Weekday (Mo = 1)'] == 5 , 1, inplace=True)
merged_df['Placement - Weekday (Mo = 1)'].mask(merged_df['Placement - Weekday (Mo = 1)'] == 6 , 0, inplace=True)
merged_df['Placement - Weekday (Mo = 1)'].mask(merged_df['Placement - Weekday (Mo = 1)'] == 7 , 0, inplace=True)



# The average time taken from placement time to pickup time in seconds

merged_df['average_time']= (merged_df['Placement - Time'] + merged_df['Confirmation - Time'] + 
                            merged_df['Arrival at Pickup - Time'] + merged_df['Pickup - Time'])/4


#Removing Outliers
size= 21201
new_df= merged_df[:len(train_df)] 
y = new_df['Time from Pickup to Arrival']
removed_outliers = y.between(y.quantile(.05), y.quantile(.95))

print(str(y[removed_outliers].size) + "/" + str(size) + " data points remain.") 

#y[removed_outliers].plot().get_figure()
index_names = new_df[~removed_outliers].index
new_df.drop(index_names, inplace=True)


# Splitting the baseline_train DataFrame into the X and Y variable using the Pandas .iloc[] method.

#Y = merged_df[:len(train_df)][['time_from_pickup_to_arrival']]
#X = merged_df[:len(train_df)].drop('time_from_pickup_to_arrival',axis=1)


# Using sklearn.model_selection, train_test_split() method to split the baseline_X and baseline_Y.
# Test size will be 0.2 (20% of the data will the test case).

#Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    #X, Y, 
    #test_size=0.2, 
    #random_state=21)


# train test split
y = new_df[['Time from Pickup to Arrival']]
x_train = new_df.drop('Time from Pickup to Arrival',axis=1)
x_test = merged_df[len(train_df):].drop('Time from Pickup to Arrival',axis=1)

for col_to_delete in x_test.columns[~x_test.columns.isin(train_df.columns)]:
    del x_test[col_to_delete]


for ting in train_df.columns[~train_df.columns.isin(x_test.columns)]:
    x_test[ting] = 0

train_df = train_df.reindex_axis(sorted(train_df.columns), axis=1)

x_test = x_test.reindex_axis(sorted(x_test.columns), axis=1)

#x_train = x_train.as_matrix()
#x_test = x_test.as_matrix()

model= xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
model.fit(x_train , y)

import pickle

model_save_path = "xgb2_model.pkl"
with open(model_save_path,'wb') as file:
    pickle.dump(model,file)




