import windows
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input,Output
import pandas as pd

run = False

if run == True:
    fileNames2 = []
    directory = input("Input directory: ")
    fileNames = input("Input CSV File Names Comma Separated: ").split(",")
    for name in fileNames:
        fileNames2.append(name.replace(".csv", "-P.csv"))
    for i in range(len(fileNames)):
        dir = directory+fileNames[i]
        df = windows.DataFrameWin(csv_file=dir)
        df.modify("format_dates", "timestamp")
        df.interpolate_missing_rows()
        df.create_lag(3, "value")
        print("created dataframe "+str(i))

        training = windows.TrainingWin(df)
        training.split(["hourOfDay", "dayOfWeek", "monthOfYear", "lag1", "lag2", "lag3"], "value", 0.2)
        training.hyper_param = {
            'bootstrap': False,
            'max_depth': None,
            'max_features': 'sqrt',
            'min_samples_leaf': 1,
            'min_samples_split': 6,
            'n_estimators': 1600
        }
        training.generate_model()
        print("generated model "+str(i))

        dir2 = directory+(fileNames[i].replace(".csv",".sav"))
        training.save_model(dir2)
        training.predict_csv(directory+fileNames[i-1], "timestamp", "value", directory+fileNames2[i-1], 3)
        print("predicted csv "+str(i-1)+" using "+str(i))

app = dash.Dash()

app.layout = html.Div([
    html.H1(children='Analysis of Anomalous Energy Readings with Random Forest Regression Training Model'),
    html.H3(children="Select an option:"),
    dcc.RadioItems(id="option-select", options=['Display Heatmap', 'Display Scatter'], value='Display Heatmap'),
    html.H3(children='Enter CSV File Directory:'),
    dcc.Dropdown(["Hail-210123993-FebToMay.csv", "Qassem-210121180-FebToMay.csv", "Riyadh1-210123140-FebToMay.csv", "Riyadh2-210123971-FebToMay.csv", "Riyadh2-210123976-FebToMay.csv"], "Hail-210123993-FebToMay.csv", id='file-select'),
    dcc.Graph(id="output-graph"),
])

dfW = None
gW = None

@app.callback(
    Output(component_id='output-graph', component_property='figure'),  # callback decorator (output): output-graph:children
    [Input(component_id='option-select', component_property='value'),
    Input(component_id='file-select', component_property='value')])  # callback decorator (input): file-select:contents
def update(option, fileDir):
    global dfW
    global gW
    fileDir = "Data\\" + fileDir
    if option == "Display Heatmap":
        dfW = windows.DataFrameWin(csv_file=fileDir, parse_dates=["timestamp"])
        dfW.modify("format_dates", "timestamp")
        gW = windows.GraphWin(dfW)
        return gW.create_graph("heatmap", "value", None)
    elif option == "Display Scatter":
        dfW = windows.DataFrameWin(csv_file=fileDir.replace(".csv", "-P.csv"), parse_dates=["timestamp"])
        df = dfW.df
        df["anomaly"] = df["anomaly"].astype(str)
        return px.scatter(df, x="timestamp", y="value", color="anomaly", color_discrete_sequence=["red", "blue"])
app.run_server(debug=False)