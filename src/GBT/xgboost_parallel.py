from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from time import time
import argparse

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd

parser = argparse.ArgumentParser(description="Logistic Regression.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--N', type=int, default=20, help='Level of parallelism/number of threads')
parser.add_argument('--data', default='../../data/Clean_data.csv', help='Input file containing all features and labels, used to train a logistic model')

args = parser.parse_args()
data_path = args.data


raw_df=pd.read_csv(data_path)
train_df,test_df=train_test_split(raw_df,test_size=0.3,random_state=2)
test_df,val_df=train_test_split(test_df,test_size=0.5,random_state=2)

print('train_df.shape : ',train_df.shape)
print('val_df.shape : ',val_df.shape)
print('test_df.shape : ',test_df.shape)


input_cols=['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']
target_cols='Diabetes_binary'

train_inputs=train_df[input_cols].copy()
train_targets=train_df[target_cols].copy()

val_inputs=val_df[input_cols].copy()
val_targets=val_df[target_cols].copy()

test_inputs=test_df[input_cols].copy()
test_targets=test_df[target_cols].copy()

train_inputs.to_csv('train_x.csv')  
train_targets.to_csv('train_y.csv')

test_inputs.to_csv('test_x.csv')
test_targets.to_csv('test_y.csv')


## nthreads > 7 start to take longer
model=XGBClassifier(random_state=27,learning_rate=0.35,max_depth=6,max_leaves=4,n_estimators=20,max_bin=8, min_child_weight=6,booster='gbtree', nthread = args.N)

start = time()
model.fit(train_inputs,train_targets)
end = time() - start

print("Training time : ", str(end))

print("PREDICTION for training")
preds=model.predict(train_inputs)
#print(preds)
print(accuracy_score(train_targets, preds))
print(precision_score(train_targets, preds))
print(recall_score(train_targets, preds))

print("PREDICTION for Validation")
preds=model.predict(val_inputs)
#print(preds)
print(accuracy_score(val_targets, preds))
print(precision_score(val_targets, preds))
print(recall_score(val_targets, preds))

print("PREDICTION for testing")
preds=model.predict(test_inputs)
#print(preds)
print(accuracy_score(test_targets, preds))
print(precision_score(test_targets, preds))
print(recall_score(test_targets, preds))