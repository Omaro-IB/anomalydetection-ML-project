from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from itertools import product

def random_forest_regressor(X_train, y_train, hyper_param={'bootstrap': True,'max_depth': None,'max_features': 1.0,'min_samples_leaf': 1,'min_samples_split': 2,'n_estimators': 100}):
    """
    Creates Random Forest Regressor model
    :params X_train, y_train: Pandas Dataframes; dataframe for each train/test/x/y split
    :param hyper_param: Dict: bootstrap,max_depth,max_features,min_samples_leaf,min_samples_split,n_estimators
    :return: Random forest model
    """
    clf = RandomForestRegressor(bootstrap=hyper_param["bootstrap"], max_depth=hyper_param["max_depth"],
                                max_features=hyper_param["max_features"], min_samples_leaf=hyper_param["min_samples_leaf"],
                                min_samples_split=hyper_param["min_samples_split"], n_estimators=hyper_param["n_estimators"])
    clf.fit(X_train, y_train)
    return clf

def accuracy(test_set, prediction_set):  # OK
    """
    Finds a model's accuracy
    :param test_set: The test set
    :param prediction_set: The prediction set
    :return: Float , Accuracy
    """
    try:
        return metrics.accuracy_score(test_set, prediction_set)
    except ValueError:
        return metrics.mean_absolute_error(test_set, prediction_set)

def tune_hp_RFR(X_train, y_train, X_test, y_test, param_grid):
    x1 = []
    for key in param_grid:
        x1.append(param_grid[key])
    x = []
    for xs in product(*x1):
        x.append(xs)

    def func(x):
        clf = random_forest_regressor(X_train, y_train, {'bootstrap':x[0], 'max_depth':x[1], 'max_features':x[2], 'min_samples_leaf':x[3], 'min_samples_split':x[4], 'n_estimators':x[5]})
        return accuracy(clf.predict(X_test), y_test)

    returnDict = {}
    keys = list(param_grid)
    hyper = tune_hp_rec(func, x)
    for i in range(len(keys)):
        returnDict[keys[i]] = hyper[i]

    return returnDict

def tune_hp_rec(func, x):
    start = func(x[0])
    end = func(x[-1])
    print(x, start, end)

    if len(x) == 1:
        return x[0]
    else:
        if start < end:
            return tune_hp_rec(func, x[:len(x) // 2])
        else:
            return tune_hp_rec(func, x[len(x) // 2:])
