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

def create_lag_df(df, column, lag):
    """
    Creates a Pandas dataframe with a lag column
    :param df:Pandas dataframe, with no index column, two columns of the same size
    :param column:String, The column name to be lagged
    :param lag:String, The other column name to be replaced with the  column
    :return:Pandas DataFrame, with the lag column replaced with the column but lagged (shifted up by 1)
    """
    df[lag] = df[column]
    df = df.rename(columns={column: 'lag2', lag: "lag1"})
    df["lag1"] = df["lag1"].shift(-1)
    return df[:-1]

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # transform a time series dataset into a supervised learning dataset
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

# def walk_forward_validation(train, test):
#     predictions = list()
#     # seed history with training dataset
#     history = [x for x in train]
#     # step over each time-step in the test set
#     for i in range(len(test)):
#         # split test row into input and output columns
#         testX, testy = test[i, :-1], test[i, -1]
#         # fit model on history and make a prediction
#         yhat = train.random_forest_forecast(history, testX)
#         # store forecast in list of predictions
#         predictions.append(yhat)
#         # add actual observation to history for the next loop
#         history.append(test[i])
#         # summarize progress
#         print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
#     # estimate prediction error
#     error = mean_absolute_error(test[:, -1], predictions)
#     return error, test[:, 1], predictions