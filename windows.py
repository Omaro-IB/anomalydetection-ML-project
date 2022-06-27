import data_exploration
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import warnings

import graphing
import train


class Window:
    def __init__(self):
        pass

class DataFrameWin(Window):
    """
    DataFrame Window Object; manages all functions related to DataFrames including displaying the window
    """

    def __init__(self, df=None, csv_file=None, drop_columns=None, parse_dates=None):
        if df is not None:
            self.df = df
        elif csv_file is not None:
            self.df = data_exploration.initialize_df(csv_file, drop=drop_columns, parse_dates=parse_dates)
        else:
            raise ValueError("At least one of 'df' or 'init_directory' must be initialized")

        self.dates_formatted = False
        self.show_all = False
        self.nlags = 0

        super().__init__()

    def modify(self, mode, value):
        # mode = "drop" ^ "format_dates"
        if mode == "drop":
            if isinstance(value, list):
                for i in value:
                    self.df.drop(i, axis=1, inplace=True)
            else:
                self.df.drop(value, axis=1, inplace=True)
        elif mode == "format_dates":
            timeCols = data_exploration.format_time(self.df, value)
            insertLoc = self.df.columns.get_loc(value)

            cols = (list(self.df.columns))
            del cols[insertLoc]
            cols = cols[:insertLoc] + ["hourOfDay", "dayOfWeek", "monthOfYear"] + cols[insertLoc:]

            newDF = pd.DataFrame(columns=cols)
            for i in newDF.columns:
                if i in self.df:
                    newDF[i] = self.df[i]
                else:
                    newDF[i] = timeCols[i]
            self.df = newDF
            self.dates_formatted = True
        else:
            raise ValueError("Invalid mode- valid options: 'drop','format_dates'")

    def create_lag(self, n, col):
        for lag in range(1,n+1):
            self.df["lag%s" % lag] = (self.df[col].shift(lag))[lag:]
        self.nlags+=n

    def interpolate_missing_rows(self):
        self.df = self.df.interpolate()

    def split_by_col(self, col_name):
        """
        Splits itself into different Data Frames organized in a dictionary
        :param col_name:The column name to split by
        :return:Dictionary of DataFrameWin instances
        """
        grouped = data_exploration.group_by_n(self.df, col_name)
        grouped2 = {}
        if self.dates_formatted:
            for key in grouped:
                tempDF = DataFrameWin(df=grouped[key])
                tempDF.dates_formatted = True
                grouped2[key] = tempDF
        else:
            for key in grouped:
                grouped2[key] = DataFrameWin(df=grouped[key])
        return grouped2

    def __str__(self):
        if self.show_all:
            self.show_all = False
            with pd.option_context('display.max_columns', None):
                return self.df.__str__()
        return self.df.__str__()

    def __repr__(self):
        if self.show_all:
            self.show_all = False
            with pd.option_context('display.max_columns', None):
                return self.df.__repr__()
        return self.df.__repr__()


class GraphWin(Window):
    def __init__(self, DataFrameWin):
        super().__init__()
        self.DataFrameWin = DataFrameWin

    def export_DF(self):
        return self.DataFrameWin

    def create_graph(self, mode, x, y):
        # mode = "missing-values" ^ "pacf" ^ "heatmap"
        if mode == "missing-values":
            # x = x-axis, y = frequency
            plt.hist(data_exploration.create_missing_value_histogram_data(self.DataFrameWin.df, x, y), bins=50)
            plt.show()
        elif mode == "pacf":
            # x = x axis, y = x-scale (int)
            plot_pacf(self.DataFrameWin.df.dropna()[x], lags=y)
            plt.show()
        elif mode == "heatmap":
            # x = value column, y = None
            if not self.DataFrameWin.dates_formatted:
                raise ValueError("Incompatible DataFrameWin with this graph- dates must be formatted first- try "
                                 "DataFrameWin.modify('format_dates', time_col)")
            else:
                graphing.graph_heatmap(self.DataFrameWin.df, "dayOfWeek", "hourOfDay", x)
        else:
            raise ValueError("Invalid mode- valid options: 'missing-values', 'pacf', 'heatmap'")


class TrainingWin(Window):
    def __init__(self, DataFrameWin):
        super().__init__()
        self.DataFrameWin = DataFrameWin
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.hyper_param = None
        self.model = None

    def export_DF(self):
        return self.DataFrameWin

    def split(self, x_attrs, y_attr, test_size):
        newDF = self.DataFrameWin.df
        newDF.dropna(inplace=True)
        X = newDF[x_attrs]  # Features
        y = newDF[y_attr]  # Label
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)

    def tune_hps(self, param_grid):  # TODO: finish this and continue from "changes.txt"
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("Data has not yet been train-test-split- try TrainingWin.split(x_attrs, y_attr, test_size)")
        else:
            self.hyper_param = train.tune_hp_RFR(self.X_train, self.y_train, self.X_test, self.y_test, param_grid)

    def generate_model(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data has not yet been train-test-split- try TrainingWin.split(x_attrs, y_attr, test_size)")
        else:
            if self.hyper_param is None:
                self.model = train.random_forest_regressor(self.X_train, self.y_train)
            else:
                self.model = train.random_forest_regressor(self.X_train, self.y_train, hyper_param=self.hyper_param)

    def save_model(self, dir_):
        pickle.dump(self.model, open(dir_, 'wb'))

    def load_model(self, dir_):
        self.model = pickle.load(open(dir_, 'rb'))

    def predict_csv(self, dir1, timeCol, valueCol, dir2, anomaly=False):
        if isinstance(dir1, str):
            DataFrameWin_temp = DataFrameWin(csv_file=dir1)
            DataFrameWin_temp.modify("format_dates", timeCol)
            DataFrameWin_temp.interpolate_missing_rows()
            # DataFrameWin_temp.show_all=True
            # print(DataFrameWin_temp)
            # print(valueCol)
            DataFrameWin_temp.create_lag(self.DataFrameWin.nlags, valueCol)

            y_preds = [None for i in range(self.DataFrameWin.nlags)]
            anomalies = [None for i in range(self.DataFrameWin.nlags)]
            groundTruthCol = np.asarray(DataFrameWin_temp.df[valueCol])
            std = DataFrameWin_temp.df[valueCol].std()
            colNames = ["hourOfDay", "dayOfWeek", "monthOfYear"]
            for i in range (1, self.DataFrameWin.nlags+1):
                colNames.append("lag" + str(i))
            predDF = DataFrameWin_temp.df[colNames]

            for i, j in predDF.iterrows():
                currRow = (np.asarray(j))
                if not (True in pd.isna(currRow)):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        currResult = self.model.predict(currRow.reshape(1, len(currRow)))
                    y_preds.append(currResult[0])
                    if anomaly:
                        groundtruth = groundTruthCol[len(y_preds)-1]
                        if abs(currResult[0]-groundtruth) > anomaly * abs(std):
                            anomalies.append(1)
                        else:
                            anomalies.append(0)

            exportDF = DataFrameWin_temp.df
            exportDF["prediction_results"] = y_preds
            if anomaly:
                exportDF["anomaly"] = anomalies
            exportDF.to_csv(dir2)

    def predict_list(self, lis):
        return self.model.predict(np.array(lis).reshape(1, len(lis)))
