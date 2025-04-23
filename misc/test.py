# app.py
import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

data = pd.DataFrame({
    "Date": pd.date_range("2024-01-01", periods=10),
    "Sales": [100, 120, 90, 130, 160, 150, 140, 180, 200, 190]
})

fig = px.line(data, x="Date", y="Sales", title="Sales Over Time")

app.layout = html.Div([
    html.H1("Dashboard"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)