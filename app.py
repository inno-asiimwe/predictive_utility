import base64
import datetime
import io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot
import statsmodels
import statsmodels.api as sm 
import statsmodels.formula.api as smf
import itertools
import warnings
import plotly.plotly as py
import plotly.tools as tls

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table_experiments as dt

import pandas as pd
import numpy as np 

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Butabika Prediction System'),

    html.Div(children='''
    A utility system to Forecast number of patients at Butabika Hospital for the next five years
    '''),

    dcc.Upload(
        id ='upload-data',
        children=html.Div(
            [
                'Drag and Drop or ',
                html.A('Select a csv File')
            ]
        ),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'})
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            dateparse = lambda dates: pd.datetime.strptime(dates, '%y')
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8'))
            )
            ts = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),
                parse_dates=['year'],
                index_col='year'
            )

            ts = ts.sort_index()
            
            p = d = q = range(0, 2)
            pdq = list(itertools.product(p, d, q))
            seasonal_pdq = [(x[0], x[1], x[2], 5) for x in list(itertools.product(p, d, q))]

            optimal_parameters = get_optimal_parameters(ts, pdq, seasonal_pdq)

            mod = optimal_model(ts, optimal_parameters)

            predictions = mod.get_forecast(steps=5)


        else:
            return html.Div(
                ['File must be in specified format']
            )
    except Exception as e:
        print(e)
        return html.Div(
            [
                'There was an error processing this file try again'
            ]
        )

    return html.Div(
        [
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),
            dt.DataTable(rows=df.to_dict('records')),
            html.Div(
                str(predictions.predicted_mean)
            )
        ]
    )


def get_optimal_parameters(ts, pdq, spdq):
    warnings.filterwarnings("ignore")
    lowest_aic = 10000
    for param in pdq:
        for param_seasonal in spdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(
                    ts,
                    order=param,
                    seasonal_order=param_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = mod.fit()
                print('ARIMA{}x{}5 - AIC:{}'.format(param, param_seasonal, results.aic))
                if lowest_aic > results.aic:
                    lowest_aic = results.aic
                    optimal_pdq = param
                    optimal_seasonal_pdq = param_seasonal
            except:
                continue

    return {'pdq': optimal_pdq, 'spdq': optimal_seasonal_pdq}


def plot_prediction(series, model_prediction):
    mpl_fig = pyplot.figure()
    prediction = model_prediction.get_forecast(steps=5)
    prediction_ci = prediction.conf_int()
    ax = series.plot(label='observed', figsize=(20, 15))
    prediction.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(
        prediction_ci.index,
        prediction_ci.iloc[:,0],
        prediction_ci.iloc[:,1], color='k', alpha=.25)
    ax.set_xlabel('year')
    ax.set_ylabel('patients')
    ax = mpl_fig.add_subplot(ax)
    mpl_fig.savefig()

    plotly_figure = tls.mpl_to_plotly(ax)

    return plotly_figure


def optimal_model(series, optimal_params):
    mod = sm.tsa.statespace.SARIMAX(
        series,
        order=optimal_params['pdq'],
        seasonal_order=optimal_params['spdq'],
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return mod.fit()


@app.callback(
    Output('output-data-upload', 'children'),
    [
        Input('upload-data', 'contents'),
        Input('upload-data', 'filename'),
        Input('upload-data', 'last_modified')
    ]
)
def update_output(content, filename, date):
    if content:
        children = [
            parse_contents(content, filename, date)
        ]
        return children

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)