# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from typing import Callable
import dash
from dash.html.Main import Main
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.long_callback import DiskcacheLongCallbackManager
from dash.exceptions import PreventUpdate

import time
import diskcache

import plotly.express as px
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime

from Common import Logger
from Common import Optimization
from StateMachine import StateMachine
from StateMachine import DoE

from Dashboard import Layout
from distutils.dir_util import copy_tree

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
    ]
)
def processCallback(set_progress, n_clicks):
    global processPauseFlag

    maxStr = "4"
    resume()

    dateString = datetime.now().strftime("%d%m%Y_%H.%M.%S")
    backUpFolder = "./Logs/ServerBackup_{}".format(dateString)
    copy_tree(Logger.getCurrentLogFolder(), backUpFolder)

    if n_clicks is None or n_clicks <= 0: return ["READY"]
    
    Logger.logInfo("Start StateMachine with InitDoE")
    mainSM = StateMachine.StateMachine(DoE.InitDoE())
    for state in mainSM: 
        set_progress(("1", maxStr, f"{state}"))
        hanldePausing(set_progress, "1", maxStr)
        

    Logger.logInfo("Find optimum")
    set_progress(("2", maxStr, f"{state}"))
    optimum = Optimization.optimizationFromDoEResult(state.result())
    Logger.logInfo("Optimum @: {}".format(optimum))


    Logger.logInfo("Start DoE around optimum")
    Logger.appendToLogFolder("DoE_Around_Optimum")
    mainSM = StateMachine.StateMachine(DoE.InitDoE(optimum=optimum))
    for state in mainSM:
        set_progress(("3", maxStr, f"{state}"))
        hanldePausing(set_progress, "3", maxStr)
        
    resume()
    return f"READY"

def hanldePausing(set_progress, minStr, maxStr):
    while isPausing(): 
        set_progress((minStr, maxStr, "PAUSING :)"))
        time.sleep(1)

@dashboard.callback(
    inputs=Input("buttonPause", "n_clicks"),
    output=Output("buttonPause", "children"),
)
def pauseCallback(n_clicks):
    if isPausing(): 
        resume() 
    else: 
        pause()

    return "Pause / Resume"

pausePath = path = Path('./PUASE.dsk')
def pause():
    if not isPausing(): path.touch()

def resume():
    if isPausing(): path.unlink()

def isPausing():
    return path.is_file()



if __name__ == "__main__":
    resume()

    Logger.initLogging()
    Logger.logInfo("Start main DoE program")
    np.set_printoptions(suppress=True)

    startServer()