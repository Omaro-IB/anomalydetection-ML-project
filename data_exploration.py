import pandas as pd

def initialize_df(dir_, index=None, drop=None, parse_dates=None):  # initializes the data-frame
    """
    Initializes Pandas dataframe from csv file
    :param dir_: directory of csv file
    :param index:The index column, default is None (will simply use no index column)
    :param drop:List of column names to drop, default is None
    :param parse_dates:List of column names with lists in them to parse, default is None
    :return:Pandas Dataframe
    """
    if index is None:
        if parse_dates is None:
            df = pd.read_csv(dir_)
        else:
            df = pd.read_csv(dir_, parse_dates=parse_dates)
    else:
        if parse_dates is None:
            df = pd.read_csv(dir_, index_col=index)
        else:
            df = pd.read_csv(dir_, index_col=index, parse_dates=parse_dates)
    if drop is not None:
        for d in drop:
            df = df.drop(d, axis=1)
    return df

def create_missing_value_histogram_data(df, col, freq_col):
    """
    Creates a list with repeating column values n times, where n = the number of times that value has NaN in freq_col
    :param df: Data frame with two columns- one x-axis column and one frequency column
    :param col: The x-axis column
    :param freq_col:The frequency column
    :return:List of column values * number of NaN's in freq_col
    """
    df = df[df[freq_col].isnull()]  # drops all non-NaN rows
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

def format_time(df, col):
    """
    Takes Pandas time series and returns a DataFrame with 3 columns: hourOfDay,dayOfWeek,monthOfYear
    :param df:Pandas DataFrame time series
    :param col: time column to format
    :return:Pandas DataFrame with 3 columns: hourOfDay, dayOfWeek, monthOfYear
    """
    returnDF = pd.DataFrame({"hourOfDay": [], "dayOfWeek": [], "monthOfYear": []})
    returnDF['hourOfDay'] = pd.to_datetime(df[col]).dt.hour
    returnDF['dayOfWeek'] = pd.to_datetime(df[col]).dt.dayofweek
    returnDF['monthOfYear'] = pd.to_datetime(df[col]).dt.month

    return returnDF
