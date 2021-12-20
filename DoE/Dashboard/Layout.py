from dash import html, Dash
import plotly.express as px

class Layout():

    def __init__(self) -> None:
        pass

    def getLayout(self):
        return html.Div(
            children=[
                html.H1(
                    style={
                        'textAlign': 'center'
                    },
                    children="Base Layout"
                )
            ]
        )

    def appendExternalCSSFiles(self, dashboard:Dash):
        pass # dashboard.css.append_css({"external_url": ""})

class DoEDashboard(Layout):

    def appendExternalCSSFiles(self, dashboard: Dash):
        dashboard.css.append_css({"external_url": "./Assets/main.css"})

    def getLayout(self):

        return html.Div(
            id="content",
            children=[
                html.H1(children="DoE Dashboard"),
                html.Div(children=[
                    html.P(style={'fontWeight': 'bold'},children="STATE:"),
                    html.P(children="waiting")
                ]
                )
            ]
        )