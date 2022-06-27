from windows import Window
from windows import DataFrameWin
from windows import GraphWin
from windows import TrainingWin


df = DataFrameWin(csv_file=r"C:\Users\omaro\Desktop\Data\Riyadh2-210123971-FebToMay.csv")  # create dataframe from CSV

df.modify("format_dates", "timestamp")  # format "timestamp" column as dates
df.modify("drop", ["city","device","vertical","attribute"])  # drop "city","device","vertical", and "attribute" columns

# Creates graphing window and graphs heatmap
graphing_window = GraphWin(df)
graphing_window.create_graph("heatmap", "value", None)
