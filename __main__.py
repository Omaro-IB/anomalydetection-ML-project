import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import data_exploration
from statsmodels.graphics.tsaplots import plot_pacf
from datetime import datetime
import plotly.express as px
import seaborn as sns

run = "heatmap"

#sets up a data-frame with timestamp as index & only meter-reading column
data_frame = data_exploration.initialize_df("train.csv", index = "timestamp", drop=["anomaly", "building_id"])
hist_data = data_frame["meter_reading"].tolist()

#sets up a data-frame with numeric index & only meter-reading column
if run == "interpolate" or run == "pacf" or run == "heatmap":
    data_frame_indexed = data_exploration.initialize_df("train.csv", drop = ["anomaly","building_id", "timestamp"])
    data_frame_interpolated = data_frame_indexed.interpolate() #interpolate
if run == "interpolate":
    #export interpolated dataframe to csv
    data_frame_interpolated.to_csv("train_inter.csv")

#sets up a data-frame with numeric index & only anomaly column
if run == "anomaly-histogram":
    data_frame_indexed_anomaly = data_exploration.initialize_df("train.csv", drop=["building_id", "meter_reading", "timestamp"])
    hist_data_anomaly = data_frame_indexed_anomaly["anomaly"].tolist()

#Prints ratio
if run == "ratio":
    number_of_non_anomalies, number_of_anomalies = hist_data_anomaly.count(0), hist_data_anomaly.count(1)  # finds ratio of anomalies
    anomalies_to_non_ratio = number_of_anomalies/number_of_non_anomalies
    print(anomalies_to_non_ratio)

#pACF chart
PACFseries = data_frame_indexed

if run == "pacf":
    #creates lag column (time_column)
    # dates = data_frame.index.tolist()
    # dates_f = []
    # for date in dates:
    #     dates_f.append(datetime.strptime(date[2:], '%y-%m-%d %H:%M:%S')-datetime(2016,1,1,0,0))
    # time_series = pd.Series(dates_f)
    # time_column = time_series.dt.days
    time_column = data_exploration.create_time_series(data_frame, lag=True)

    #only uses the first 100,000 because whole data set takes too long to run
    PACFseries.drop(list(range(100000,len(PACFseries))), axis = 0, inplace = True)
    time_column.drop(list(range(100000,len(time_column))), axis = 0, inplace = True)
    plot_pacf(PACFseries["meter_reading"], lags = time_column)
    plt.show()

if run == "heatmap":
    time_column = data_exploration.create_time_series(data_frame[0:len(data_frame):500])
    timeLists = [
        [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]]
    for time in range(len(time_column)):
        timeLists[time_column[time].weekday()][time_column[time].hour].append(data_frame["meter_reading"].iloc[time])
    heatmap_data = [[],[],[],[],[],[],[]]
    for i in range(len(timeLists)): #i = day of week
        for j in range(len(timeLists[i])): #j = time of day
            try:
                myList = [x for x in timeLists[i][j] if np.isnan(x) == False]
                avg = sum(myList) / len(myList)
                heatmap_data[i].append(avg)
            except ZeroDivisionError:
                heatmap_data[i].append("No Data")
    heatmap_data = np.array(heatmap_data)
    heatmap_dataframe = pd.DataFrame(heatmap_data)
    fig = px.imshow(heatmap_dataframe, text_auto=True)
    fig.show()


#plot meter reading histogram
if run == "histogram":
    plt.hist(hist_data, bins = 50)
    plt.show()

#plot anomaly histogram
if run == "anomaly-histogram":
    plt.hist(hist_data_anomaly, bins = 2)
    plt.show()
