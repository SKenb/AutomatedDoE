from dash import html, Dash
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
                    html.P(id="state", style={'fontWeight': 'bold'}, children="READY"),
                    html.Br(), 
                    html.Progress(id="stateProgessBar"),
                    html.Div(
                        id="buttonContainer",
                        children=[
                            html.Button(id="buttonStart", children="Let's do some DoE"),
                            html.Button(id="buttonStop", children="Coffe break - now"),
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