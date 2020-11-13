'''
Column Legend-+
"timestamp" - timestamp field for grouping the data - (We remove this one later on)
"cnt" - the count of a new bike shares
"t1" - real temperature in C
"t2" - temperature in C "feels like"
"hum" - humidity in percentage
"windspeed" - wind speed in km/h
"weathercode" - category of the weather
      1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity
      2 = scattered clouds / few clouds
      3 = Broken clouds
      4 = Cloudy
      7 = Rain/ light Rain shower/ Light rain
      10 = rain with thunderstorm
      26 = snowfall
      94 = Freezing Fog
"isholiday" - boolean field - 1 holiday / 0 non holiday
"isweekend" - boolean field - 1 if the day is weekend
"season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.
"Date" - Only the date from the old timestamp column
"Time" - Only the time from the old timestamp column
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


bikes = pd.read_csv("london_merged.csv")
df = bikes.copy()
#Looking through the data

print(bikes.columns)

print("Describing dataset:")
print(bikes.describe(include='all'))
print()

print("First five of each column:")
print(bikes.head())
print()

print("Last five of each column:")
print(bikes.tail())
print()

print("random sampling of five values:")
print(bikes.sample(5))
print()

#collecting and removing null values
nullvals = pd.isnull(bikes)
pd.set_option('display.max_columns',100)
print(nullvals.describe())

# Changing celsius to fahrenheit
def to_fahrenheit(x):
    return x * (9/5) + 32

bikes['t1'] = bikes['t1'].apply(to_fahrenheit) #changes temps to fahrenheit for the real temp
bikes['t2'] = bikes['t2'].apply(to_fahrenheit) #changes "feels like" temp to fahrenheit
print(bikes.t1)
print(bikes.t2)

bikes[['Date','Time']] = bikes.timestamp.str.split(expand=True)
bikes = bikes.drop('timestamp', axis=1)

## normalizing and scaling the data
scaler = MinMaxScaler()
## norm_df = pd.DataFrame(scaler.fit_transform(bikes), index=bikes.index, columns = bikes.columns)