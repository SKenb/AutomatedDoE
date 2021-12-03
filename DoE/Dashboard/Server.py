# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash.html.Main import Main
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
from dash.long_callback import DiskcacheLongCallbackManager

import time
import Layout
import diskcache

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

dashboard = dash.Dash(__name__, long_callback_manager=long_callback_manager)

def startServer(debug=True):

    dashboard.title = "Automated DoE"

    #dashboard._favicon
    #layout.appendExternalCSSFiles(self.dashboard)

    dashboard.layout = Layout.getDoELayout()

    dashboard.run_server(debug=debug)


@dashboard.long_callback(
    output=Output("state", "children"),
    inputs=Input("buttonStart", "n_clicks"),
    running=[
        (Output("buttonStart", "disabled"), True, False),
        (Output("buttonStop", "disabled"), False, True),
        (
            Output("state", "style"),
            {"visibility": "hidden"},
            {"visibility": "visible"},
        ),
        (
            Output("stateProgessBar", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        ),
    ],
    cancel=[Input("buttonStop", "n_clicks")],
    progress=[Output("stateProgessBar", "value"), Output("stateProgessBar", "max")],
)
def callback(set_progress, n_clicks):
    total = 10
    for i in range(total):
        time.sleep(0.5)
        set_progress((str(i + 1), str(total)))
    return [f"Clicked {n_clicks} times"]



if __name__ == "__main__":
    startServer()