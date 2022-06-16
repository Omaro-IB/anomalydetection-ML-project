import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def initialize_df(dir, index = None, drop = [], parse_dates = []): #initializes the data-frame
    if index == None:
        if parse_dates == []:
            df = pd.read_csv(dir)
        else:
            df = pd.read_csv(dir,parse_dates=parse_dates)
    else:
        if parse_dates == []:
            df = pd.read_csv(dir,index_col = index)
        else:
            df = pd.read_csv(dir,index_col = index,parse_dates=parse_dates)
    for d in drop:
        df = df.drop(d, axis=1)
    return df

def create_time_series(df, lag=False):
    dates = df.index.tolist()
    dates_f = []
    if lag:
        for date in dates:
            dates_f.append(datetime.strptime(date[2:], '%y-%m-%d %H:%M:%S')-datetime(2016,1,1,0,0))
            time_series = pd.Series(dates_f)
            time_column = time_series.dt.days
    else:
        for date in dates:
            dates_f.append(datetime.strptime(date[2:], '%y-%m-%d %H:%M:%S'))
            time_column = pd.Series(dates_f)
    return time_column