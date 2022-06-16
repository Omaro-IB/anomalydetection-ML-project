import data_exploration
import graphing

# DATA
data_frame = data_exploration.initialize_df("train.csv", index="timestamp", drop=["anomaly",
                                                                                  "building_id"])  # Timestamp index data-frame (Only meter-reading)
data_frame_indexed = data_exploration.initialize_df("train.csv", drop=["anomaly", "building_id",
                                                                       "timestamp"])  # Indexed data-frame (Only meter_reading)
data_frame_indexed_anomaly = data_exploration.initialize_df("train.csv", drop=["building_id", "meter_reading",
                                                                               "timestamp"])  # Indexed data-frame (meter-reading and anomaly)

# PACF Data
PACFseries = data_frame_indexed[0:len(data_frame_indexed):500] #series data
shortened_data_frame = data_frame[0:len(data_frame):500] #time data

# INTERPOLATE DATA AND EXPORT TO "train_inter.csv"
data_frame_interpolated = data_frame_indexed.interpolate()  # interpolate
data_frame_interpolated.to_csv("train_inter.csv")

# GRAPHING
# graphing.graph_histogram(data_frame)  # graph regular distribution histogram of meter readings
# print(graphing.graph_anomaly_histogram(data_frame_indexed_anomaly))  # print anomaly ratio + graph anomaly distribution
# graphing.graph_pacf(PACFseries, shortened_data_frame)  # graphs PACF graph
# graphing.graph_heatmap(data_frame)
