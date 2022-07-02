import pandas as pd
import numpy as np
import plotly.express as px


def graph_heatmap(df, dayOfWeek, hourOfDay, value):
    """
    Takes data-frame and plots heatmap (on value column) based on days of week/time of day
    :param df: Pandas data-frame with 3 columns: dayOfWeek, hourOfDay, value
    :param dayOfWeek: name of weekday column
    :param hourOfDay: name of hour column
    :param value: name of value column
    :return:None
    """
    # time_column = data_exploration.create_time_series(df[0:len(df):500])
    # time_column = data_exploration.create_time_series(df, timestamp)
    time_lists = [
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]]
    for i in range(len(df[value])):
        time_lists[df[dayOfWeek].iloc[i]][df[hourOfDay].iloc[i]].append(df[value].iloc[i])
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
    return fig
