from dash import html, Dash, dcc
import plotly.express as px
from datetime import datetime

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
                    html.P(children="STATE: "),
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
                                id='yaxis-column',
                                options=[{'label': i, 'value': i} for i in getOptions()],
                                value='Life expectancy at birth, total (years)'
                            ),
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
    return []