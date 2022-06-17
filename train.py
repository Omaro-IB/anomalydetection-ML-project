import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


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

def random_forest_forecast(data):
    def train_test_split(data, n_test):
        return data[0:n_test], data[n_test:len(data)]

    train, testx = train_test_split(data, int(len(data)*0.8))
    print(train)
    print(testx)

    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(testx[:, :3])
    return yhat[0]