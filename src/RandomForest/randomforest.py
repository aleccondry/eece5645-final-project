

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn import metrics

filepath = '/content/Training_data.csv'
data = pd.read_csv(filepath)
data.info()

y = data['Diabetes_binary']
x = data.drop(['Diabetes_binary'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

pred = rf.predict(x_train)
rf_Training_Accuracy = metrics.accuracy_score(y_train, pred)
rf_Training_Precision = metrics.precision_score(y_train, pred)
rf_Training_Recall = metrics.recall_score(y_train, pred)
print(rf_Training_Accuracy, rf_Training_Precision, rf_Training_Recall)

pred = rf.predict(x_test)
rf_Test_Accuracy = metrics.accuracy_score(y_test, pred)
rf_Test_Precision = metrics.precision_score(y_test, pred)
rf_Test_Recall = metrics.recall_score(y_test, pred)
print(rf_Test_Accuracy, rf_Test_Precision, rf_Test_Recall)

filepath = '/content/Test_data.csv'
data_test = pd.read_csv(filepath)
y_validate = data_test['Diabetes_binary']
x_validate = data_test.drop(['Diabetes_binary'], axis=1)

pred = rf.predict(x_validate)

rf_Accuracy = metrics.accuracy_score(y_validate, pred)
rf_Precision = metrics.precision_score(y_validate, pred)
rf_Recall = metrics.recall_score(y_validate, pred)
print(rf_Accuracy, rf_Precision, rf_Recall)