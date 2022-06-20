import data_exploration
import graphing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import metrics


#   ALL WEEKS
# DATA
data_frame = data_exploration.initialize_df("train.csv")  # Dataframe with everything- no index
grouped_df = data_exploration.group_by_n(data_frame, "building_id")  # Dictionary of DF's grouped by building_id
BUILDING_ID = 1241  # 1278


#   WEEK 1
# INTERPOLATE DATA AND EXPORT TO "train_inter.csv"
grouped_df_inter = {}
for key in grouped_df:
    x = grouped_df[key].interpolate()
    grouped_df_inter[key] = x
    # x.to_csv("interpolated-by-building_id\\"+str(key)+"_inter.csv") #exports to csv (only needs to run once)


#   WEEK 2
# TRAINING: Random Forest Classifier
# df_RFC = data_exploration.create_RF_data(grouped_df_inter[BUILDING_ID].dropna(), 0, "meter_reading")  # Create proper RFC data
# X=df_RFC[['hourOfDay', 'dayOfWeek', 'monthOfYear', 'meter_reading']]  # Features
# y=df_RFC['anomaly']  # Labels
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80% training and 20% test
# clf=RandomForestClassifier()  # Create a Gaussian Classifier
# clf.fit(X_train,y_train)  # Train the model using the training sets y_pred=clf.predict(X_test)
# y_pred=clf.predict(X_test)  # Predict
# print(y_pred)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))  # Model Accuracy, how often is the classifier correct?

# TRAINING: Random Forest Regressor
df_RFR = data_exploration.create_RF_data(grouped_df_inter[BUILDING_ID].dropna(), 0, "meter_reading")  # Create proper RFR data
x = df_RFR.iloc[:, :-1]
y = df_RFR.iloc[:, -1:]
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(x, y)
# The Data Features to Predict from (hour, day, month, and meter_reading) of that particular BUILDING_ID
features = [22, 2, 1, 359.805]
Y_pred = regressor.predict(np.array(features).reshape(1, 4))  # test the output by changing values
if Y_pred < 0.5:
    print("Not Anomaly ("+str(100-(Y_pred[0]*100))+"% confidence)")
else:
    print("Anomaly ("+str(Y_pred[0]*100)+"% confidence)")



# GRAPHING
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