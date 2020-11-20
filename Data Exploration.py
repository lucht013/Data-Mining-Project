'''
Column Legend-+
"timestamp" - timestamp field for grouping the data - (We remove this one later on)
"cnt" - the count of a new bike shares
"t1" - real temperature in C
"t2" - temperature in C "feels like"
"hum" - humidity in percentage
"wind_speed" - wind speed in km/h
"weather_code" - category of the weather
      1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity
      2 = scattered clouds / few clouds
      3 = Broken clouds
      4 = Cloudy
      7 = Rain/ light Rain shower/ Light rain
      10 = rain with thunderstorm
      26 = snowfall
      94 = Freezing Fog
"is_holiday" - boolean field - 1 holiday / 0 non holiday
"is_weekend" - boolean field - 1 if the day is weekend
"season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.
"Date" - Only the date from the old timestamp column
"Time" - Only the time from the old timestamp column
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from dmba import regressionSummary, AIC_score, backward_elimination, stepwise_selection, forward_selection

# from sklearn.preprocessing import StandardScaler, MinMaxScaler


bikes = pd.read_csv("london_merged.csv")
df = bikes.copy()
# Looking through the data

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

# Collecting and removing null values
nullvals = pd.isnull(bikes)
pd.set_option('display.max_columns', 100)
print(nullvals.describe())


# Changing celsius to fahrenheit
def to_fahrenheit(x):
    return x * (9 / 5) + 32


bikes['t1'] = bikes['t1'].apply(to_fahrenheit)  # changes temps to fahrenheit for the real temp
bikes['t2'] = bikes['t2'].apply(to_fahrenheit)  # changes "feels like" temp to fahrenheit
print(bikes.t1)
print(bikes.t2)

bikes[['Date', 'Time']] = bikes.timestamp.str.split(expand=True)
bikes = bikes.drop('timestamp', axis=1)

# changing km/h to mph


def to_mph(x):
    return x * 0.621371


bikes['wind_speed'] = bikes['wind_speed'].apply(to_mph)  # changes the windspeed to mph

"""
do heatmap for the correlation.
cnt vs temp, windspeed, 
humidity vs its high corrs values (temp, wind_speed, weather_code)
"""

corr = bikes.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, fmt=".1f", cmap='RdBu', annot=True)
plt.title('Bike Sharing Correlation Matrix')
plt.show()

# training data/building Linear regression
excluded = ('Date', 'Time', 'cnt')
predictors = [s for s in bikes.columns if s not in excluded]
# predictors = ['hum']
outcome = 'cnt'
X = bikes[predictors]
y = bikes[outcome]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

model = LinearRegression()
model.fit(train_X, train_y)

train_pred = model.predict(train_X)
train_results = pd.DataFrame({'cnt': train_y, 'predicted': train_pred, 'residual': train_y - train_pred})
print("And here are the results!")
print("")
print(train_results.head())

# scatter plots

df.plot.scatter(x='cnt', y='t1', legend=False)
plt.show()

df = bikes[['cnt', 't1', 't2', 'wind_speed', 'hum', 'weather_code']]
axes = scatter_matrix(df, alpha=0.5, figsize=(6, 6), diagonal='kde')
corr = df.corr().values
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("%.3f" % corr[i, j], (0.8, 0.8),
                        xycoords='axes fraction', ha='center', va='center')
plt.show()

# histogram
ax = bikes.hum.hist()
ax.set_xlabel('Humidity')
ax.set_ylabel('Count')
plt.title('Humidity vs Count Histogram')
plt.show()

ax = bikes.t1.hist()
ax.set_xlabel('Temperature (in F)')
ax.set_ylabel('Count')
plt.title('Temperature vs Count Histogram')
plt.show()

ax = bikes.wind_speed.hist()
ax.set_xlabel('Wind Speed')
ax.set_ylabel('Count')
plt.title('Wind Speed vs Count Histogram')
plt.show()

# line graphs
cntTime = bikes[['cnt', 'Time']]  # just a df with only time and cnt

x = ['00:00:00', '02:00:00', '04:00:00', '06:00:00', '08:00:00', '10:00:00', '12:00:00', '14:00:00', '16:00:00', '18:00:00', '20:00:00', '22:00:00', '24:00:00']
x_ticks = np.arange(0, 24, 2)
byHour = cntTime.groupby(by=[cntTime.Time]).mean()
ax = byHour.plot(ylim=[0, 3200], legend=True)
ax.set_xlabel('Hour')
ax.set_ylabel('Count (mean)')
plt.tight_layout()
plt.title('Average Bike Usage by Hour')
plt.show()

bikes.Date = pd.to_datetime(bikes.Date, format='%Y/%m/%d')
bikes_ts = pd.Series(bikes.cnt.values, index=bikes.Date)
bikes_ts.plot()
plt.xlabel('Year')
plt.ylabel('Total bikes shares')
plt.title('Bike Usage Over Two Years')
plt.show()

bike_LM = LinearRegression()
bike_LM.fit(train_X, train_y)
regressionSummary(valid_y, bike_LM.predict(valid_X))








def train_model(variables):
    if len(variables) == 0:
        return None
    model = LinearRegression()
    model.fit(train_X[list(variables)], train_y)
    return model


def score_model(model, variables):
    if len(variables) == 0:
        return AIC_score(train_y, 
                         [train_y.mean()] * len(train_y), 
                         model, df=1)
    return AIC_score(train_y, 
                     model.predict(train_X[variables]), model)

    
allVars = train_X.columns

best_model, best_vars = backward_elimination(allVars, train_model, score_model, verbose=True)

best_model1, best_vars1 = stepwise_selection(allVars, train_model, score_model, verbose=True)
    
best_model2, best_vars2 = forward_selection(allVars, train_model, score_model, verbose=True)


## REGULARIZATION
bayR = BayesianRidge(normalize=True)
bayR.fit(train_X, train_y)
regressionSummary(valid_y, bayR.predict(valid_X))
alpha = bayR.lambda_ / bayR.alpha_
print('Bayesian ridge chosen Regularization: ', alpha)
