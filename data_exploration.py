import pandas as pd
from datetime import datetime


def initialize_df(dir: object, index: object = None, drop: object = [], parse_dates: object = []) -> object:  # initializes the data-frame
    """
    Initializes Pandas dataframe from csv file
    :param dir: directory of csv file
    :param index:The index column, default is None (will simply use no index column)
    :param drop:List of column names to drop, default is empty list []
    :param parse_dates:List of column names with lists in them to parse, default is empty list []
    :return:Pandas Dataframe
    """
    if index == None:
        if parse_dates == []:
            df = pd.read_csv(dir)
        else:
            df = pd.read_csv(dir, parse_dates=parse_dates)
    else:
        if parse_dates == []:
            df = pd.read_csv(dir, index_col=index)
        else:
            df = pd.read_csv(dir, index_col=index, parse_dates=parse_dates)
    for d in drop:
        df = df.drop(d, axis=1)
    return df


def create_time_series(df, lag=False):
    """
    Creates a non-indexed Pandas Dataframe with either lag values as integers (from first date in days) or just the
    raw datetime object
    :param df:Pandas dataframe with the index being the timestamps with format %y-%m-%d %H:%M:%S
    :param lag:Boolean value, determines whether the return dataframe is lag or raw, default is False
    :return:Pandas Dataframe with lag values or raw datetime objects
    """
    dates = df.index.tolist()
    dates_f = []
    if lag:
        for date in dates:
            first_date = datetime.strptime(df.index[0][2:], "%y-%m-%d %H:%M:%S")
            dates_f.append(datetime.strptime(date[2:], '%y-%m-%d %H:%M:%S') - first_date)
            # dates_f.append(datetime.strptime(date[2:], '%y-%m-%d %H:%M:%S') - datetime(2016, 1, 1, 0, 0))
            time_series = pd.Series(dates_f)
            time_column = time_series.dt.days
    else:
        for date in dates:
            dates_f.append(datetime.strptime(date[2:], '%y-%m-%d %H:%M:%S'))
            time_column = pd.Series(dates_f)
    return time_column
