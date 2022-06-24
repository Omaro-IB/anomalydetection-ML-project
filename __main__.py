# import data_exploration
# import graphing
# import train
# import anomaly_predictor
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from difflib import SequenceMatcher
import anomaly_predictor
import tkinter as tk
import tkinter.filedialog
from os import listdir
import pickle

BUILDING_ID = 1278  # 1278
listOfModels = []
listOfFiles = listdir("models")
print("Loading models...")
for file_ in listOfFiles:
    listOfModels.append(pickle.load(open("models\\"+file_, 'rb')))
print("Loaded models")

root = tk.Tk()

def func():
    n = int(nEntry.get())
    root.destroy()
    for file in Q:
        anomaly_predictor.processCSV(file, "timestamp", "value", listOfModels, n)
def askopenfile(Q):
    Q.append(tk.filedialog.askopenfile(mode='r').name)
    queue.delete(0,tk.END)
    for i in Q:
         queue.insert(tk.END, i)

Q = []

tk.Label(text="Queue").grid(row=1, column=1)
tk.Label(text="Number of random models to use: ").grid(row=1, column=2)

nEntry = tk.Entry()
nEntry.grid(row = 1, column=3)

queue = tk.Listbox()
queue.grid(row=2,column=1)

addToQueue = tk.Button(root, text='Add to Queue', command=lambda: askopenfile(Q))
addToQueue.grid(row=3,column=1)

submitButt = tk.Button(root, text="Submit", command=func)
submitButt.grid(row=2, column=2)

root.mainloop()

if __name__ == "__main__":
    pass
    # print(anomaly_predictor.predictor(22,2,1,360,"models",10))

    # Interpolate Data and Export to "train_inter.csv" (only needs to run once)
    # grouped_df_inter = train.interpolate_data_split("building_id", "train.csv", dir2 = "interpolated-by-building_id")

    # Interpolate Data but don't export
    # grouped_df_inter = train.interpolate_data_split("building_id", "train.csv")
    # features = [22, 2, 1, 175.492, 170.0445, 164.597, 160.181]  # [hourOfDay, dayOfWeek, monthOfYear, meter_reading, lag1, lag2, lag3]


    # TRAINING: Random Forest Classifier (tuned hyper-parameters)
    # X_test, y_test, RFC_model = train.random_forest_classifier(grouped_df_inter[BUILDING_ID], hyper_param={'n_estimators': 2000, 'bootstrap': False, 'max_depth': None, 'max_features': 1.0, 'min_samples_leaf': 1, 'min_samples_split': 2})
    # print(train.predict(RFC_model, features, show=True))
    # acc = train.accuracy(y_test, RFC_model.predict(X_test))
    # print(acc)


    # TRAINING: Random Forest Regressor (tuned hyper-parameters) + 3 lag columns
    # X_test, y_test, RFR_model = train.random_forest_regressor(grouped_df_inter[BUILDING_ID], hyper_param={'n_estimators': 2000, 'bootstrap': False, 'max_depth': None, 'max_features': 1.0, 'min_samples_leaf': 1, 'min_samples_split': 2})
    # print(train.predict(RFR_model, features, show=True))
    # X_test_predicted = RFR_model.predict(X_test)
    # #Calculating Accuracy
    # testSetPositives = [i for i in range(len(np.asarray(y_test))) if np.asarray(y_test)[i] >= 0.5]
    # predicSetPositives = [i for i in range(len(np.asarray(X_test_predicted))) if np.asarray(X_test_predicted)[i] >= 0.5]
    # predicSetNegatives = [i for i in range(len(np.asarray(X_test_predicted))) if np.asarray(X_test_predicted)[i] < 0.5]
    # testSetNegatives = [i for i in range(len(np.asarray(y_test))) if np.asarray(y_test)[i] < 0.5]
    # sm = SequenceMatcher(None, testSetPositives, predicSetPositives)
    # sm2 = SequenceMatcher(None, testSetNegatives, predicSetNegatives)
    # positives_accuracy = sm.ratio()
    # negatives_accuracy = sm2.ratio()
    # print("Positives Accuracy: {} \t Negatives Accuracy {}".format(positives_accuracy, negatives_accuracy))

    # acc = train.accuracy(y_test, X_test_predicted)



    # CREATE MODELS FOR USE WITH ANOMALY PREDICTOR (takes ~2 hours to run)
    # for i in grouped_df_inter:
    #      X_test, y_test, RFR_model = train.random_forest_regressor(grouped_df_inter[i].dropna(), hyper_param={'n_estimators': 2000, 'bootstrap': False, 'max_depth': None, 'max_features': 1.0, 'min_samples_leaf': 1, 'min_samples_split': 2})
    #      # print(train.predict(RFR_model, features, show=True))
    #      train.save_model(RFR_model, "models\\model"+str(i))


    # TUNING HYPER-PARAMETERS (takes hours to run)
    # def create_model(hp):
    #     x, y, z = train.random_forest_regressor(grouped_df_inter[BUILDING_ID], hyper_param=hp)
    #     return z
    #
    # def create_model2(hp):
    #     x, y, z = train.random_forest_classifier(grouped_df_inter[BUILDING_ID], hyper_param=hp)
    #     return z

    # param_grid = {
    #     'bootstrap': [True, False],
    #     'max_depth': [80, 90, 100, 110, None],
    #     'max_features': [1.0, "sqrt", "log2"],
    #     'min_samples_leaf': [1, 3, 4, 5],
    #     'min_samples_split': [2, 8, 10, 12],
    #     'n_estimators': [2000]
    # }
    # print(train.tune_hp(create_model, features, 1, param_grid))
    # print(train.tune_hp(create_model2, features, 1, param_grid))


    # GRAPHING
    # data_frame = data_exploration.initialize_df("train.csv")  # Dataframe for graphing
    # Histogram
    # graphing.graph_histogram(data_frame)  # graph bar chart, x-axis = building ids, y-axis = no. of NaNs

    # Anomaly Graph
    # print(graphing.graph_anomaly_histogram(data_frame_indexed_anomaly))  # print anomaly ratio + graph anomaly distribution

    # PACF Graph
    # graphing.graph_pacf(grouped_df_inter[BUILDING_ID].dropna(), 8)  # graphs PACF graph

    # Heatmap
    # heatmap_df = grouped_df_inter[BUILDING_ID]
    # heatmap_df = heatmap_df.set_index("timestamp")
    # graphing.graph_heatmap(heatmap_df)
