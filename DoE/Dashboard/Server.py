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
from dash.exceptions import PreventUpdate

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
        (Output("buttonPause", "disabled"), False, True),
        (
            Output("state", "style"),
            {"visibility": "hidden"},
            {"visibility": "visible"},
        ),
        (
            Output("stateProgessBar", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        )
    ],
    cancel=[Input("buttonStop", "n_clicks")],
    progress=[
        Output("stateProgessBar", "value"), 
        Output("stateProgessBar", "max"), 
        Output("processState", "children")
    ],
)
def callback(set_progress, n_clicks):
    global processPauseFlag

    if n_clicks is None or n_clicks <= 0: return ["READY"]

    total = 10
    for i in range(total):

        while processPauseFlag: 
            set_progress((str(i + 1), total, f"Pausing :D"))
            time.sleep(0.5)

        time.sleep(0.0)
        set_progress((str(i + 1), total, f"Index = {i}"))
        
        print(processPauseFlag)
    

    return [f"READY"]


@dashboard.callback(
    output=Output("buttonPause", "children"),
    inputs=[
        Input("buttonPause", "n_clicks"),
        Input("processPauseFlag", "data"),
    ]
)
def callback(n_clicks, processPauseFlag):
    processPauseFlag = not processPauseFlag

    return [
            "Resume" if processPauseFlag else "Coffe time - Pause",
            processPauseFlag
        ]



if __name__ == "__main__":
    startServer()