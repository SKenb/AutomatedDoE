# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash.html.Main import Main
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd

import Layout


def startServer(debug=True):

    dashboard = dash.Dash(__name__)
    dashboard.title = "Automated DoE"
    layout = Layout.DoEDashboard()
    #self.dashboard._favicon
    #self.layout.appendExternalCSSFiles(self.dashboard)

    dashboard.layout = layout.getLayout()

    print(dashboard.get_asset_url('main.css'))

    dashboard.run_server(debug=debug)


if __name__ == "__main__":
    startServer()