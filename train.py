import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


def split(df, train_percentage, seed=None):
    """
    Takes pandas Dataframe and splits it into two other dataframes: train and test
    :param df:Pandas Dataframe
    :param train_percentage:Percentage of data to be used as training data
    :param seed:Seed to be used to generate reproducible random data, if None then no seed is used, default is None
    :return:A 2-tuple with train split as the first element and the test split as the second
    """
    if seed is None:
        msk = np.random.rand(len(df)) < train_percentage
    else:
        msk = np.random.RandomState(seed).rand(len(df)) < train_percentage
    return df[msk], df[~msk]

def standardize_data(df, column):
    """
    Takes pandas Dataframe and standardizes it to normal distribution
    :param df: Pandas Dataframe
    :param column: String, column name to standardize
    :return: Standardized Pandas Dataframe (has zero mean and unit variance)
    """
    x = df.copy()
    col = np.asarray(x[[column]])
    col = preprocessing.StandardScaler().fit(col).transform(col)
    x[column] = col
    return x
