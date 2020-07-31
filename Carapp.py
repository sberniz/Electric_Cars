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
from joblib import dump

electric_cars = pd.read_csv('Electric_Vehicle_Population_Data.csv')

electric_cars['Base MSRP'] = electric_cars['Base MSRP'].replace({0:np.NaN,845000:np.NaN})
electric_cars.dropna(subset=['Base MSRP'], inplace=True ) #drop NA values for base MSRP
#electric_cars = electric_cars.drop(11853) #Removing Problematic rows
#electric_cars = electric_cars.drop(25326)
#Will use make, model, and model year might add other features
#Will do a randomsplit test
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
app_pipe = make_pipeline(
    OrdinalEncoder(),
    XGBRegressor(
    n_estimators = 4000,
    max_depth=7,
    learning_rate=0.5,
    n_jobs=-1,
    random_state=42
    )
)
app_pipe.fit(X_train,y_train)
y_pred = app_pipe.predict(X_test)
print("R2:",r2_score(y_test,y_pred))
print("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))

row = X_test.iloc[[10]]
y_single = app_pipe.predict(row)
print(y_single)

def predictor(df):
    X = df.copy()
    y_pred_final = app_pipe.predict(X)
    return y_pred_final
y_fin = predictor(row)
print(y_fin)
print(row)

dump(app_pipe, 'app_pipe.joblib', compress=True)