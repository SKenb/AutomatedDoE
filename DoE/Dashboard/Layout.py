from dash import html, Dash, dcc
import plotly.express as px
from datetime import datetime
import os

def getDoELayout():

    return html.Div(
        id="content",
        children=[
            html.H1(
                children="DoE Dashboard"
            ),
            html.Div(
                id="processControl",
                children=[
                    html.P(id="state", children="READY"),
                    html.P(id="processState", children=""),
                    html.Br(), 
                    html.Progress(id="stateProgessBar"),
                    html.Div(
                        id="buttonContainer",
                        children=[
                            html.Button(id="buttonStart", children="Let's do some DoE"),
                            html.Button(id="buttonPause", children="Pause / Resume"),
                        ]
                    ),
                    html.Br(), html.Br(), 
                    html.Button(
                        id="buttonStop", 
                        children="STOP",
                        style={"width": "80%", "margin": "auto"}
                    ),
                    dcc.Store(id='processPauseFlag'),
                    html.Div(
                        id="intermidiateResults",
                        children=[
                            html.Hr(),
                            dcc.Dropdown(
                                id='plotResultsDropDown',
                                options=[{'label': i, 'value': i} for i in getOptions()],
                                value='None'
                            ),
                            html.Img(
                                id="plotResultsImage"
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                id="footer",
                children=[
                    html.A(children="Made with Plotly", href="https://plotly.com/", target="_blank"),
                    html.A(children="RCPE", href="https://www.rcpe.at/", target="_blank"),
                    html.A(children="Sebastian Knoll", href="https://sebastianknoll.net/me", target="_blank")
                ]
            ),
            html.Div(
                id="header",
                children=[
                    html.P(children="System Time:"),
                    html.P(id="systemTime", children=datetime.now().strftime("%d.%m.%Y - %H:%M:%S"))
                ]
            )
        ]
    )

def getOptions():
    filelist=os.listdir('./assets/DoE/')
    for fichier in filelist[:]:
        if not(fichier.endswith(".png")):
            filelist.remove(fichier)

    filelist2=os.listdir('./assets/DoE/DoE_Around_Optimum/')
    for fichier in filelist2[:]:
        if not(fichier.endswith(".png")):
            filelist2.remove(fichier)

    filelist.extend(["DoE_Around_Optimum/" + f for f in filelist2])

    return filelist