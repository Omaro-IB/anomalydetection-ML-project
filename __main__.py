import data_exploration
import graphing
import train
import anomaly_predictor
import matplotlib.pyplot as plt
import numpy as np
BUILDING_ID = 1241  # 1278

if __name__ == "__main__":
    print(anomaly_predictor.predictor(22,2,1,360,"models",10))

    # Interpolate Data and Export to "train_inter.csv" (only needs to run once)
    # grouped_df_inter = train.interpolate_data_split("building_id", "train.csv", dir2 = "interpolated-by-building_id")

    # Interpolate Data but don't export
    # grouped_df_inter = train.interpolate_data_split("building_id", "train.csv")
    # features = [22, 2, 1, 360]  # [hourOfDay, dayOfWeek, monthOfYear, meter_reading]


    # TRAINING: Random Forest Classifier (tuned hyper-parameters)
    # X_test, y_test, RFC_model = train.random_forest_classifier(grouped_df_inter[BUILDING_ID], hyper_param={'n_estimators': 2000, 'bootstrap': False, 'max_depth': None, 'max_features': 1.0, 'min_samples_leaf': 1, 'min_samples_split': 2})
    # print(train.predict(RFC_model, features, show=True))
    # acc = train.accuracy(y_test, RFC_model.predict(X_test))
    # print(acc)


    # TRAINING: Random Forest Regressor (tuned hyper-parameters)
    # X_test, y_test, RFR_model = train.random_forest_regressor(grouped_df_inter[BUILDING_ID], hyper_param={'n_estimators': 2000, 'bootstrap': False, 'max_depth': None, 'max_features': 1.0, 'min_samples_leaf': 1, 'min_samples_split': 2})
    # print(train.predict(RFR_model, features, show=True))
    # acc = train.accuracy(y_test, RFC_model.predict(X_test))
    # print(acc)


    # CREATE MODELS FOR USE WITH ANOMALY PREDICTOR (takes ~30 mins to run)
    # for i in grouped_df_inter:
    #     X_test, y_test, RFR_model = train.random_forest_regressor(grouped_df_inter[i].dropna(), hyper_param={'n_estimators': 2000, 'bootstrap': False, 'max_depth': None, 'max_features': 1.0, 'min_samples_leaf': 1, 'min_samples_split': 2})
    #     print(train.predict(RFR_model, features, show=True))
    #     train.save_model(RFR_model, "models\\model"+str(i))


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
    #     'n_estimators': [100, 200, 1000, 2000]
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
