import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import data_exploration
from statsmodels.graphics.tsaplots import plot_pacf
import plotly.express as px


def graph_anomaly_histogram(df):
    """
    Takes dataframe and plots anomaly distribution histogram
    :param df:Indexed pandas dataframe with 2 columns: meter-reading and anomaly
    :return:
    """
    hist_data_anomaly = df["anomaly"].tolist()
    number_of_non_anomalies, number_of_anomalies = hist_data_anomaly.count(0), hist_data_anomaly.count(
        1)  # finds ratio of anomalies
    anomalies_to_non_ratio = number_of_anomalies / number_of_non_anomalies

    plt.hist(hist_data_anomaly, bins=2)
    plt.show()

    return anomalies_to_non_ratio


def graph_histogram(df):
    """
    Takes dataframe and plots histogram
    :param df: Pandas Dataframe with 1 column: meter_reading
    :return:None
    """
    hist_data = df["meter_reading"].tolist()
    plt.hist(hist_data, bins=50)
    plt.show()


def graph_pacf(series, time_data):
    """
    Takes series and lag column and plots Partial Auto-correlation graph
    :param series:Pandas dataframe with only 1 column: meter readings
    :param time_data:Pandas dataframe with the index being the timestamps with format %y-%m-%d %H:%M:%S
    :return:None
    """
    lag = data_exploration.create_time_series(time_data, lag=True)  # lag column data

    plot_pacf(series["meter_reading"], lags=lag)
    plt.show()


def graph_heatmap(df):
    """
    Takes data-frame and plots heatmap based on days of week/time of day
    :param df: Pandas data-frame with two columns: timestamp (index column) and meter-reading
    :return:None
    """
    time_column = data_exploration.create_time_series(df[0:len(df):500])
    time_lists = [
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]]
    for time in range(len(time_column)):
        time_lists[time_column[time].weekday()][time_column[time].hour].append(df["meter_reading"].iloc[time])
    heatmap_data = [[], [], [], [], [], [], []]
    for i in range(len(time_lists)):  # i = day of week
        for j in range(len(time_lists[i])):  # j = time of day
            try:
                my_list = [x for x in time_lists[i][j] if not np.isnan(x)]
                avg = sum(my_list) / len(my_list)
                heatmap_data[i].append(avg)
            except ZeroDivisionError:
                heatmap_data[i].append("No Data")
    heatmap_data = np.array(heatmap_data)
    heatmap_dataframe = pd.DataFrame(heatmap_data)
    fig = px.imshow(heatmap_dataframe, text_auto=True)
    fig.show()
