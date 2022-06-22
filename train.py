import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import data_exploration
import pickle


# def split(df, train_percentage, seed=None):
#     """
#     Takes pandas Dataframe and splits it into two other dataframes: train and test
#     :param df:Pandas Dataframe
#     :param train_percentage:Percentage of data to be used as training data
#     :param seed:Seed to be used to generate reproducible random data, if None then no seed is used, default is None
#     :return:A 2-tuple with train split as the first element and the test split as the second
#     """
#     if seed is None:
#         msk = np.random.rand(len(df)) < train_percentage
#     else:
#         msk = np.random.RandomState(seed).rand(len(df)) < train_percentage
#     return df[msk], df[~msk]
#
# def standardize_data(df, column):
#     """
#     Takes pandas Dataframe and standardizes it to normal distribution
#     :param df: Pandas Dataframe
#     :param column: String, column name to standardize
#     :return: Standardized Pandas Dataframe (has zero mean and unit variance)
#     """
#     x = df.copy()
#     col = np.asarray(x[[column]])
#     col = preprocessing.StandardScaler().fit(col).transform(col)
#     x[column] = col
#     return x


def interpolate_data_split(col, dir1, dir2=""):
    """
    Interpolates missing values in csv file
    :param col: String, name of column to split csv files by
    :param dir1: String, directory to csv file- "path\to\file.csv"
    :param dir2: String, directory to save files- default = "" (if left empty, then it doesn't export)
    :return:Dictionary, all interpolated dataframes with keys being the split column
    """
    data_frame = data_exploration.initialize_df(dir1)
    grouped_df = data_exploration.group_by_n(data_frame, col)
    grouped_df_inter = {}

    for key in grouped_df:
        x = grouped_df[key].interpolate()
        grouped_df_inter[key] = x
        if dir2 != "":
            x.to_csv(dir2+"\\"+str(key)+"_inter.csv")  # exports to csv

    return grouped_df_inter

def random_forest_classifier(df, hyper_param={'bootstrap': True,'max_depth': None,'max_features': 'sqrt','min_samples_leaf': 1,'min_samples_split': 2,'n_estimators': 100}):
    """
    Creates Random Forest Classifier model
    :param df: Pandas DataFrame
    :param hyper_param: Dict: bootstrap,max_depth,max_features,min_samples_leaf,min_samples_split,n_estimators
    :return:X_test, y_test, random forest model
    """
    df_RFC = data_exploration.create_RF_data(df, 0, "meter_reading")  # Create proper RFC data
    X = df_RFC[['hourOfDay', 'dayOfWeek', 'monthOfYear', 'meter_reading']]  # Features
    y = df_RFC['anomaly']  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 80% training and 20% test
    clf = RandomForestClassifier(bootstrap=hyper_param["bootstrap"], max_depth=hyper_param["max_depth"],
                                max_features=hyper_param["max_features"], min_samples_leaf=hyper_param["min_samples_leaf"],
                                min_samples_split=hyper_param["min_samples_split"], n_estimators=hyper_param["n_estimators"])
    clf.fit(X_train, y_train)  # Train the model using the training sets y_pred=clf.predict(X_test)

    return X_test, y_test, clf

def random_forest_regressor(df, hyper_param={'bootstrap': True,'max_depth': None,'max_features': 1.0,'min_samples_leaf': 1,'min_samples_split': 2,'n_estimators': 100}):
    """
    Creates Random Forest Regressor model
    :param df: Pandas DataFrame
    :param hyper_param: Dict: bootstrap,max_depth,max_features,min_samples_leaf,min_samples_split,n_estimators
    :return: X_test, y_test, random forest model
    """
    df_RFR = data_exploration.create_RF_data(df, 0, "meter_reading")  # Create proper RFR data
    X = df_RFR[['hourOfDay', 'dayOfWeek', 'monthOfYear', 'meter_reading']]  # Features
    y = df_RFR['anomaly']  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 80% training and 20% test
    # x = df_RFR.iloc[:, :-1]
    # y = df_RFR.iloc[:, -1:]
    clf = RandomForestRegressor(bootstrap=hyper_param["bootstrap"], max_depth=hyper_param["max_depth"],
                                max_features=hyper_param["max_features"], min_samples_leaf=hyper_param["min_samples_leaf"],
                                min_samples_split=hyper_param["min_samples_split"], n_estimators=hyper_param["n_estimators"])
    clf.fit(X_train, y_train)
    return X_test, y_test, clf


def save_model(model, dir_):
    dir_ = dir_+".sav"
    pickle.dump(model, open(dir_, 'wb'))

def predict(model, features, show=False):
    """
    Predicts using given features
    :param model: The model to predict with
    :param features: List, with 4 elements of features
    :param show: Boolean, If True, print anomaly/not anomaly with % confidence, default = False
    :return: Y_pred, value between 0 (not anomaly) and 1 (anomaly)
    """
    Y_pred = model.predict(np.array(features).reshape(1, 4))  # test the output by changing values
    if show:
        if Y_pred < 0.5:
            print("Not Anomaly ("+str(100-(Y_pred[0]*100))+"% confidence)")
        else:
            print("Anomaly ("+str(Y_pred[0]*100)+"% confidence)")
    return Y_pred

def accuracy(test_set, prediction_set):
    """
    Finds a model's accuracy
    :param test_set: The test set
    :param prediction_set: The prediction set
    :return: Float , Accuracy
    """
    return metrics.accuracy_score(test_set, prediction_set)

def tune_hp(func, features, target, param_grid):
    """
    Tunes the hyperparamaters of RFC or RFR model
    :param func: Function to create the model
    :param features: List: 4 elements to test accuracy with
    :param target: Float, target value to be as close as possible
    :param param_grid: Dict with keys being the hyper-params and value being list of values to try
    :return:The best hyper-paramaters
    """
    history = []
    history2 = []
    # bootstrap_n, max_depth_n, max_features_n, min_samples_leaf_n, min_samples_split_n, n_estimators_n = len(param_grid["bootstrap"]), len(param_grid["max_depth"]), len(param_grid["max_features"]), len(param_grid["min_samples_leaf"]), len(param_grid["min_samples_split"]), len(param_grid["n_estimators"])
    order = ["n_estimators", "bootstrap", "max_depth", "max_features", "min_samples_leaf", "min_samples_split"]
    for a in param_grid[order[0]]:
        for b in param_grid[order[1]]:
            for c in param_grid[order[2]]:
                for d in param_grid[order[3]]:
                    for e in param_grid[order[4]]:
                        for f in param_grid[order[5]]:
                            history.append({order[0]:a, order[1]:b, order[2]:c, order[3]:d, order[4]:e, order[5]:f})
                            history2.append(predict(func({order[0]:a, order[1]:b, order[2]:c, order[3]:d, order[4]:e, order[5]:f}), features)[0])

    print(history2)
    print(history)
    closest = history2[0]
    closestt = history[0]
    for i in range(1,len(history)):
        if abs(history2[i]-target) < abs(closest-target):
            closest = history2[i]
            closestt = history[i]

    return closestt




