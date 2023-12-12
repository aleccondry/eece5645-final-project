import tensorflow as tf
import pandas as pd
import numpy as np
print("TensorFlow version:", tf.__version__)

data = pd.read_csv('../../data/Clean_data.csv')

print(data.head())

train_data = data.sample(frac=0.75, random_state=1)

test_data = data.drop(train_data.index)

max_val = train_data.max(axis= 0)
min_val = train_data.min(axis= 0)
 
range = max_val - min_val
train_data = (train_data - min_val)/(range)
 
test_data =  (test_data- min_val)/range

X_train = train_data.drop('Diabetes_binary',axis=1)
X_test = test_data.drop('Diabetes_binary',axis=1)
y_train = train_data['Diabetes_binary']
y_test = test_data['Diabetes_binary']
 
# We'll need to pass the shape
# of features/inputs as an argument
# in our model, so let's define a variable 
# to save it.
input_shape = [X_train.shape[1]]
 
print(input_shape)

model = tf.keras.Sequential([
 
    tf.keras.layers.Dense(units=64, activation='relu',
                          input_shape=input_shape),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
model.summary()

model.compile(optimizer='adam',  
               metrics=['accuracy', 'Precision', 'Recall'],
              # MAE error is good for
              # numerical predictions
              loss='mae') 

losses = model.fit(X_train, y_train,
                    
                   # it will use 'batch_size' number
                   # of examples per example
                   batch_size=256, 
                   epochs=15,  # total epoch
 
                   )

print(model.predict(X_test.iloc[0:3, :]))
print(y_test.iloc[0:3])
performance = model.evaluate(X_test, y_test)

