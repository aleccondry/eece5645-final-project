import tensorflow as tf
import pandas as pd
import time
import matplotlib.pyplot as plt


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
num_hidden_layers = 64
num_epochs = 50
model = tf.keras.Sequential([
 
    tf.keras.layers.Dense(units=num_hidden_layers, activation='sigmoid',
                          input_shape=input_shape),
    tf.keras.layers.Dense(units=num_hidden_layers, activation='sigmoid'),
    tf.keras.layers.Dense(units=1)
])
model.summary()

model.compile(optimizer='adam',  
               metrics=['accuracy', 'Precision', 'Recall'],
              # MAE error is good for
              # numerical predictions
              loss='mae') 


start_time = time.time()
losses = model.fit(X_train, y_train,
                    
                   # it will use 'batch_size' number
                   # of examples per example
                   batch_size=256, 
                   epochs=num_epochs,  # total epoch
                   )
# print(model.predict(X_test.iloc[0:3, :]))
training_time = time.time() - start_time
print(f"Training time = {training_time} seconds")
plt.plot(losses.history['accuracy'])
plt.plot(losses.history['precision'])
plt.plot(losses.history['recall'])
plt.plot(losses.history['loss'])

plt.title('Model Performance for Serial NN')
plt.ylabel('Performance')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Precision', 'Recall', 'Loss'], loc='upper left')
plt.savefig(f'plots/serial_{num_hidden_layers}_neurons_{num_epochs}_epochs.png')
# plt.show()

# print(y_test.iloc[0:3])
# performance = model.evaluate(X_test, y_test)

