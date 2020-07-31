import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
##IMPORTERS
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, LogisticRegression,Ridge,RidgeCV
from scipy import stats
from category_encoders import OneHotEncoder, OrdinalEncoder
import plotly.express as px
from sklearn.preprocessing import StandardScaler


import dash
import dash_bootstrap_components as dbc

electric_cars = pd.read_csv('Electric_Vehicle_Population_Data.csv')
electric_cars['Base MSRP'] = electric_cars['Base MSRP'].replace({0:np.NaN,845000:np.NaN})
electric_cars.dropna(subset=['Base MSRP'], inplace=True ) #drop NA values for base MSRP
def wrangle1split(df):
      X = df.copy()
      y = X['Base MSRP']
      X = X[['Model Year','Make','Model','Electric Vehicle Type','Clean Alternative Fuel Vehicle (CAFV) Eligibility','Electric Range']]
      X_train,X_val,y_train,y_val = train_test_split(X,y,train_size = 0.8, test_size=0.2,random_state=42)
      return X_train, X_val, y_train, y_val
#SPlitter
X_train ,X_test, y_train, y_test = wrangle1split(electric_cars) #Train Test SPlit
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.2,train_size=0.8, random_state=42) #Train/Validation Split

#XG Boost Regression with early stopping and hyper parameter tuning, best model , will be deployment for post/apptop if the score hasn't improved in 50 rounds
#Pipeline for app
from joblib import load
app_pipe = load('assets/app_pipe.joblib')
#y_pred = app_pipe.predict(X_test)[0]
external_stylesheets = [
    dbc.themes.BOOTSTRAP, # Bootswatch theme
    'https://use.fontawesome.com/releases/v5.9.0/css/all.css', # for social media icons
    'style.css' #for slider
]

meta_tags=[
    {'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, meta_tags=meta_tags)
app.config.suppress_callback_exceptions = True # see https://dash.plot.ly/urls
app.title = 'Electric Cars Price Predictor' # appears in browser title bar
server = app.server