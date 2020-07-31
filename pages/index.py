# Imports from 3rd party libraries
import dash
import pandas as pd
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from joblib import load
app_pipe = load('assets/app_pipe.joblib')

# Imports from this application
from app import app,X_test,y_test,X_val,y_val

# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        ##Car Prices according to Make,Model, Model Year, wether or not is Clean Air Elegible, and electric Range.
        Click Button Below to get a price prediction

           """
        ),
        dcc.Link(dbc.Button('Price Predictions', color='primary'), href='/predictions')
    ],
    md=4,
)

#gapminder = px.data.gapminder()
fig = px.scatter(X_test, x="Model", y="Electric Range", color=y_test,title="Electric Car Prices Depending on Model and Electric Range")
fig.update_layout(xaxis_title="")

column2 = dbc.Col(
    [
        dcc.Graph(figure=fig),
    ]
)

layout = dbc.Row([column1, column2])