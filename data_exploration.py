import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error


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

def create_missing_value_histogram_data(df, col, freq_col):
    """
    Creates a list with repeating column values n times where n = the number of times that value has NaN in freq_col
    :param df: Data frame with two columns- one x axis column and one frequency column
    :param col: The x axis column
    :param freq_col:The frequency column
    :return:List of column values * number of NaN's in freq_col
    """
    df = df[df[freq_col].isnull()] # drops all non-NaN rows
    return df[col].tolist()

def group_by_n(df, col):
    """
    Splits a dataframe into a dictionary with each key being the values in "col" column and values being the rows
    :param df:Pandas DataFrame
    :param col:String, name of column
    :return:Dict, with keys of each value in column and values being the rows
    """
    keys = set(df[col].tolist())
    dic = {}
    for key in keys:
        dic[key] = df[df[col] == key]
    return dic

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

def create_RF_data(df, n, col):
    """
    Takes Pandas DataFrame and turns into a DataFrame ready to process by RFC algorithm
    :param df:Pandas DataFrame with timestamp column
    :param n:Number of Lag Columns
    :param col: Column to be lagged
    :return:Pandas DataFrame with no index column, hour, weekday, and month column, and n lag columns of col column
    """
    RFC_df = pd.DataFrame({"hourOfDay":[],"dayOfWeek":[],"monthOfYear":[]})
    RFC_df['hourOfDay']=pd.to_datetime(df["timestamp"]).dt.hour
    RFC_df['dayOfWeek']=pd.to_datetime(df["timestamp"]).dt.dayofweek
    RFC_df['monthOfYear']=pd.to_datetime(df["timestamp"]).dt.month


    for lag in range(n+1):
        if lag == 0:
            RFC_df[col]=df[col]
        else:
            RFC_df["lag%s" %lag]=(df[col].shift(-lag))[:-lag]

    RFC_df["anomaly"] = df["anomaly"]

    return RFC_df


    # def create_lag_df(df, column, lag):
#     """
#     Creates a Pandas dataframe with a lag column
#     :param df:Pandas dataframe, with no index column, two columns of the same size
#     :param column:String, The column name to be lagged
#     :param lag:String, The other column name to be replaced with the column
#     :return:Pandas DataFrame, with the lag column replaced with the column but lagged (shifted up by 1)
#     """
#     df[lag] = df[column]
#     df = df.rename(columns={column: 'lag2', lag: "lag1"})
#     df["lag1"] = df["lag1"].shift(-1)
#     return df[:-1]