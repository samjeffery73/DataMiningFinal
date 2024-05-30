'''
Sam Jeffery
Data Mining Final 

Weather Prediction Models
4/29/2024


META DATA:
************************************
This dataset is Recordings of daily meteorological observations from 2000 to 2010 in 18 locations(resulting in 3654 daily observations)

Locations:
Basel (Switzerland), Budapest (Hungary), Dresden, Düsseldorf, Kassel, München (all  Germany), De Bilt and Maastricht (the Netherlands), 
Heathrow (UK), Ljubljana (Slovenia),  Malmo and Stockholm (Sweden), Montélimar, Perpignan and Tours (France), Oslo (Norway), Roma (Italy), and Sonnblick (Austria). 

Variables collected are below. Some variables are available for some cities and some are not.

Variables:
‘mean temperature’, ‘max temperature’, and ‘min temperature’, 'cloud_cover', 'wind_speed', 'wind_gust', 'humidity', 'pressure', 'global_radiation', 
'sunshine' wherever those were available. 

Temperature are given in degree Celsius, wind speed and gust in m/s, humidity in fraction of 100%, sea level  pressure in 1000 hPa,
global radiation in 100 W/m2, precipitation amounts in centimeter, sunshine in hours.


From Kaggle:

C   : cloud cover in oktas
FG   : wind speed in 0.1 m/s
FX   : wind gust in 0.1 m/s
HU   : humidity in 1 %
PP   : sea level pressure in 0.1 hPa
QQ   : global radiation in W/m2
RR   : precipitation amount in 0.1 mm
SS   : sunshine in 0.1 Hours
TG   : mean temperature in 0.1 &#176;C
TN   : minimum temperature in 0.1 &#176;C
TX   : maximum temperature in 0.1 &#176;C


CONVERTED to:
| Feature (type)   | Column name 		| Description 		| Physical Unit 	|
|------------------|----------------------|-----------------------|-----------------|
| mean temperature | _temp_mean  		| mean daily temperature| in 1   	|
| max temperature  | _temp_max   		| max daily temperature | in 1   	|
| min temperature  | _temp_min   		| min daily temperature | in 1   	|
| cloud_cover      | _cloud_cover		| cloud cover           | oktas  		|
| wind_speed       | _wind_gust  		| wind gust    		| in 1 m/s 		|
| wind_gust        | _wind_speed 		| wind speed   		| in 1 m/s 		|
| humidity         | _humidity   		| humidity              | in 1 %  		|
| pressure         | _pressure   		| pressure              | in 1000 hPa  	|
| global_radiation | _global_radiation 	| global radiation      | in 100 W/m2  	|
| precipitation    | _precipitation 	| daily precipitation 	| in 10 mm  	|
| sunshine    	 | _sunshine 		| sunshine hours  	| in 0.1 hours 	|


'''


# To start, lets load the data and check out everything.
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np
import datetime as dt


df = pd.read_csv(r"weather_prediction_dataset.csv")
#print(df_unclean.head())
#print(df_unclean.info())


# Data is extremely columnar. Looking at it we can see its all similar data, but is split by "CITYNAME_" There is one city that is "CIT_YNAME_attr"
# Lets make an actual table, where each column represents a data rather than a location.
melted = df.melt(id_vars=["DATE", "MONTH"], var_name="column", value_name="value")
melted['city'] = melted['column'].str.extract(r'([A-Z]+(?:_[A-Z]+)*)')
melted.city.unique()
melted['column'] = melted['column'].str.extract(r'[A-Z]+(?:_[A-Z]+)*_(.*)')
#print(melted)
pivoted = melted.pivot(index=('DATE', 'MONTH', 'city'), columns='column', values='value').reset_index()
#print(pivoted.head())

# We can see what the meta data was talking about now with there being some cities where data wasnt collected.
# This may be due to varying resources, etc...
# This dataset looks pretty clean now. Lets drop nulls and start doing some things.
#print(pivoted.isna().groupby(pivoted['city']).sum())
cleaned = pivoted.dropna().reset_index(drop=True)

#Cleaning date and making a measurable attribute called 'day', which is just a day out of the 365 days in a year..
cleaned['DATE'] = pd.to_datetime(cleaned['DATE'], format = '%Y%m%d')
cleaned['day']= cleaned['DATE'].dt.day_of_year

# We have a date and dont really need a month anymore.
cleaned = cleaned.drop(columns='MONTH', axis = 1)
print(cleaned.describe())

# Exploring Data

# Gonna look at humidity over time.
fig, axs = plt.subplots(nrows=1, ncols=len(cleaned['city'].unique()), figsize=(15, 5))
# Time Series 
for i, city in enumerate(cleaned['city'].unique()):
    city_data = cleaned[cleaned['city'] == city]
    axs[i].plot(city_data['DATE'], city_data['humidity'], alpha=0.8, linewidth=0.4)
    axs[i].set_title(city)
    axs[i].set_xlabel('Date')
    axs[i].set_ylabel('humidity')
plt.tight_layout()
plt.show()



# Looking at temperature and humidity
plt.figure(figsize=(8, 6))
plt.scatter(cleaned['temp_mean'], cleaned['humidity'], alpha=0.5)
plt.title('Humidity vs. Temperature')
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('Humidity')
plt.grid(True)
plt.tight_layout()
plt.show()


# Box Plot of Humidity Across Cities
plt.figure(figsize=(10, 6))
cleaned.boxplot(column='humidity', by='city', grid=False)
plt.title('Humidity Across Cities')
plt.xlabel('City')
plt.ylabel('Humidity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Definitely performing regression, as we have nothing but continuous attributes.
attributes = cleaned[['temp_mean', 'day', 'cloud_cover', 'global_radiation', 'humidity', 'precipitation', 'pressure', 'sunshine', 'wind_gust', 'wind_speed']]
attributes_no_year = attributes.drop(columns='day')

# Box Plot of all data
plt.figure(figsize=(12,5))
sns.boxplot(data=attributes_no_year)
plt.show()

## Correlation matrix
selected_corr = attributes.corr()
plt.figure(figsize=(12, 12))
mask = np.triu(np.ones_like(selected_corr, dtype=bool))
sns.heatmap(selected_corr, mask=mask, robust=True, center=0, square=True, cmap="RdYlBu_r", linewidths=0.6, annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()



# Models

scaler = StandardScaler()
X = cleaned[['cloud_cover', 'dayofyear','wind_speed', 'global_radiation', 'precipitation', 'pressure', 'sunshine', 'wind_gust','temp_mean', 'city']]
y = cleaned['humidity']

# Dont want to standardize cities (we'll encode)
features_numeric = X.drop(columns =['city'])
X[features_numeric.columns] = scaler.fit_transform(features_numeric)

# Encoding and splits
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12) 


# Starting our models
DT = DecisionTreeRegressor()
DT.fit(X_train, y_train)
DT_predictions = DT.predict(X_test)
print("Decision Tree MSE:", mean_squared_error(y_test, DT_predictions), "with R2 of ", r2_score(y_test, DT_predictions))

# Plot the decision tree(Dear god this was a mistake.)
#plt.figure(figsize=(20,10))
#plot_tree(DT, filled=True, feature_names=X_train.columns)
#plt.show()


# Since there is more variance, lets try a different type of regression, like Linear.
LR = LinearRegression()
LR.fit(X_train, y_train)
LR_predictions = LR.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, LR_predictions), "with R2 of ", r2_score(y_test, LR_predictions))


# Forests, because they're good.
RF = RandomForestRegressor()
RF.fit(X_train, y_train)
RF_predictions = RF.predict(X_test)
print("Forests MSE", mean_squared_error(y_test, RF_predictions), "with R2 of ", r2_score(y_test, RF_predictions))



# We can also display our prediction test
plt.figure(figsize=(16, 8))
sns.scatterplot(x=y_test, y=RF_predictions, color='blue', label='Testing Data', alpha=.5)
sns.lineplot(x=y_test, y=y_test, color='red', label='Perfect Prediction')
sns.lineplot(x=y_test, y=RF_predictions, color='green', label='Regression Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Regression Plot')
plt.legend()
plt.show()

# Now, lets check the correlation between each variable.
importances = RF.feature_importances_
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)


