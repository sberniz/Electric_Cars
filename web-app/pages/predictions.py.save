# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from joblib import load
app_pipe = load('assets/app_pipe.joblib')
import pandas as pd
import dash_table
from collections import OrderedDict
# Imports from this application
import app
from app import app,X_test
from joblib import load
app_pipe = load('assets/app_pipe.joblib')
#from app import app
# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout

column1 =  dbc.Col(
    [
        dcc.Markdown('## Predictions', className='mb-5'),
        html.P('Select the Options below to get an estimate price for  your Electric Car'), 
        dcc.Markdown('#### Electric Range'), 
        dcc.Slider(
            id='Electric_Range', 
            min=50, 
            max=300, 
            step=1, 
            value=100, 
            marks={n: str(n) for n in range(50,350,50)}, 
            className='mb-5 slider'
        ), 
        html.Div(id='slider-output-container',style={'padding-bottom': 40}),
        dcc.Markdown('#### Model Year'), 
        dcc.Dropdown(
            id='Model_Year', 
            options = [
                {'label': i, 'value': i}
                    for i in X_test['Model Year'].unique()
            ],  
            className='mb-5 slider', 
        ),
        dcc.Markdown('#### Make'), 
        dcc.Dropdown(
            id='Make', 
            options = [
                {'label': i, 'value': i}
                    for i in X_test['Make'].unique()
            ],  
            className='mb-5', 
        ),
         dcc.Markdown('#### Model'), 
        dcc.Dropdown(
            id='Model', 
            options = [
                {'label': i, 'value': i}
                    for i in X_test['Model'].unique()
            ],  
            className='mb-5', 
        ),
        dcc.Markdown('#### Vehicle Type'), 
        dcc.Dropdown(
            id='Vehicle_Type', 
            options = [
                {'label': i, 'value': i}
                    for i in X_test['Electric Vehicle Type'].unique()
            ],  
            className='mb-5', 
        ),
        dcc.Markdown('#### Is it CAFV Elegible?'), 
        dcc.Dropdown(
            id='CAFV', 
            options = [
                {'label': i, 'value': i}
                    for i in X_test['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].unique()
            ],  
            className='mb-5', 
        ),   
    ],
    md=4,
)
column2 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## Price

            Price for Selected Features

            """
        ),
        html.Div(id='prediction-content', className='lead')
        
    ],
    md=4,
)

@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('Electric_Range', 'value')])
def update_output(value):
    return html.Strong('Electric Range:',style={'padding':0}),'{} miles'.format(value)
@app.callback(
    dash.dependencies.Output('prediction-content', 'children'),
    [dash.dependencies.Input('Model_Year', 'value'),
    dash.dependencies.Input('Make', 'value'),
    dash.dependencies.Input('Model', 'value'),
    dash.dependencies.Input('Vehicle_Type', 'value'),
    dash.dependencies.Input('CAFV', 'value'), 
    dash.dependencies.Input('Electric_Range', 'value')])
def predict(Model_Year, Make, Model, Vehicle_Type, CAFV,Electric_Range):
    df = pd.DataFrame(
        columns=['Model Year','Make','Model','Electric Vehicle Type','Clean Alternative Fuel Vehicle (CAFV) Eligibility','Electric Range'],
        data=[[Model_Year, Make, Model, Vehicle_Type, CAFV,Electric_Range]]
    )
    y_pred = app_pipe.predict(df)[0]
    print(y_pred)
    return f'{Model_Year, Make, Model, Vehicle_Type, CAFV,Electric_Range}'



layout = dbc.Row([column1, column2])
