import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


def plot_loss(history):
    plt.figure(figsize=(20,5))
    plt.plot(history.history['loss'], 'g', label='Training Loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.xlim([0, 100])
    plt.ylim([0, 300])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.waitforbuttonpress(-1)


def plot_data(x_data,y_data, x, y, title = None):
    plt.figure(figsize = (15,5))
    plt.scatter(x_data, y_data, label='Ground Truth', color='green', alpha=0.5)
    plt.plot(x, y, color='k', label='Model Predictions')
    plt.xlim([3,9])
    plt.ylim([0,60])
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Price [$K]')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.waitforbuttonpress(-1)


dataset = keras.datasets.boston_housing
feature_index = 5 #index of the average number of rooms per household, per StarLib website



#split dataset into train and test sets
(x_train, y_train), (x_test, y_test) = dataset.load_data()
print(x_train[0,feature_index], y_train[0])

#extract the feature we want to train the perceptron on (avg. number of rooms per household)
x_train_avg_rooms = x_train[:,feature_index]
x_test_avg_rooms = x_test[:,feature_index]

#create a model

#sequential, since the model consists of sequential layers of neurons
model = keras.models.Sequential()
#add input layer(shape is (1,0), since we only use one value as input)
model.add(keras.layers.Input((1,)))
#add one neuron(perceptron) as a layer. dense means every neuron of the neural network is interconnected.
model.add(keras.layers.Dense(units = 1)) 
#visualize the summary of the neural network(tunable parameters, layers, input shape etc)
model.summary()

#show the dataset with the selected feature
plt.figure(figsize=(15, 5))
plt.xlabel("Average Number of Rooms")
plt.ylabel("Median Price")
plt.grid(True)
plt.scatter(x_train_avg_rooms[:], y_train, c = "green", alpha = 0.5)
plt.waitforbuttonpress(-1)


#compile the model
model.compile(optimizer=keras.optimizers.RMSprop(0.005), loss = "mse")

#train the model
#x and y are training data; x is input and y is output.
#Batch size is how many data are passed at the same time to the model
#epochs is how many times will the process be repeated
#validation split is what percentage of train data will be used to validate the model
history = model.fit(x=x_train_avg_rooms,
                    y=y_train,
                    batch_size=16,
                    epochs=101,
                    validation_split=0.3,
                    )

#convenience function to show the training process and the loss drop during the process.
plot_loss(history)


#predict the median price of a home with [3, 4, 5, 6, 7] rooms.
x = [3, 4, 5, 6, 7]
x = tf.convert_to_tensor(x)

y_pred = model(x)
for idx in range(len(x)):
    print(f"Predicted price of a home with {x[idx]} rooms: ${int(y_pred[idx] * 10) / 10}K")


#generate feature data that spans the range of interest for the independent variable.
x = np.linspace(3, 9, 10)
print(x)

#use the model to predict the dependent variable.
y = model.predict(x)

#show the linear equation corresponding to predicted y values
plot_data(x_train_avg_rooms, y_train, x,y, "Linear Regression")


