import data_exploration
import graphing
import train
import matplotlib.pyplot as plt
import pandas as pd

#   ALL WEEKS
# DATA
data_frame = data_exploration.initialize_df("train.csv", index="timestamp", drop=["anomaly","building_id"])  # Timestamp index data-frame (Only meter-reading)
data_frame2 = data_exploration.initialize_df("train.csv", drop=["building_id"], parse_dates=["timestamp"]) #Data frame to be used for training
data_frame_RFR = data_exploration.create_lag_df(data_exploration.initialize_df("train.csv", drop=["anomaly","building_id"]), "meter_reading", "timestamp")
# data_frame_indexed = data_exploration.initialize_df("train.csv", drop=["anomaly", "building_id","timestamp"])  # Indexed data-frame (Only meter_reading)
# data_frame_indexed_anomaly = data_exploration.initialize_df("train.csv", drop=["building_id", "meter_reading","timestamp"])  # Indexed data-frame (meter-reading and anomaly)



#   WEEK 2
#TRAINING
train_df, test_df = train.split(data_frame2, 0.8, seed=192881) # train-test 80% split
#graphing.plot_scatter(train_df, "timestamp", "meter_reading") # plot data

std_train_df = train.standardize_data(train_df, "meter_reading") # standardize data
#graphing.plot_scatter(pd.DataFrame(std_train_df), "timestamp", "meter_reading") # plot standardized data

std_RFR = train.standardize_data((train.standardize_data(data_frame_RFR, "lag1")),"lag2") # standardize RFR data
sup_std_RFR = data_exploration.series_to_supervised(std_RFR) # supervise standardized RFR data

print(sup_std_RFR)
yhat = train.random_forest_forecast(sup_std_RFR[0:len(sup_std_RFR):500]) #find yhat of supervised standardized RFR data
print(yhat)


#   WEEK 1
# PACF Data
# PACFseries = data_frame_indexed[0:len(data_frame_indexed):500] #series data
# shortened_data_frame = data_frame[0:len(data_frame):500] #time data

# INTERPOLATE DATA AND EXPORT TO "train_inter.csv"
# data_frame_interpolated = data_frame_indexed.interpolate()  # interpolate
# data_frame_interpolated.to_csv("train_inter.csv")

# GRAPHING
# graphing.graph_histogram(data_frame)  # graph regular distribution histogram of meter readings
# print(graphing.graph_anomaly_histogram(data_frame_indexed_anomaly))  # print anomaly ratio + graph anomaly distribution
# graphing.graph_pacf(PACFseries, shortened_data_frame)  # graphs PACF graph
# graphing.graph_heatmap(data_frame)
