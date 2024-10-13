from dataclasses import dataclass
import seaborn as sn
import tensorflow as tf
import keras
from keras import models
from keras import datasets
from keras import layers
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

def plot_results(metrics, title=None, ylabel=None, ylim=None, metric_name=None, color=None):
    fig, ax = plt.subplots(figsize=(15, 4))

    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics,]
        metric_name = [metric_name,]

    for idx, metric in enumerate(metrics):
        ax.plot(metric, color=color[idx])

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([0, TrainingConfig.EPOCHS - 1])
    plt.ylim(ylim)
    # Tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)
    plt.show()
    plt.waitforbuttonpress(-1)

current_working_directory = os.getcwd()
print(current_working_directory)
print(current_working_directory+os.sep+"model_dropout")
#create dataclasse which will contain all of the needed parameters for the compile and fit methods
@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES:  int = 10
    IMG_HEIGHT:   int = 32
    IMG_WIDTH:    int = 32
    NUM_CHANNELS: int = 3
    
@dataclass(frozen=True)
class TrainingConfig:
    EPOCHS:        int = 31
    BATCH_SIZE:    int = 256
    LEARNING_RATE: float = 0.001 

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



SEED_VALUE = 42

#fix seed to make training deterministic.
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
cifar = datasets.cifar10

#load the dataset
(X_train, y_train), (X_test, y_test) = cifar.load_data()


#show some images from the dataset
plt.figure(figsize=(18, 8))

num_rows = 4
num_cols = 8

#plot each of the images in the batch and the associated ground truth labels.
for i in range(num_rows * num_cols):
    ax = plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(X_test[i, :, :]) #takes the ith image out of X_train dataset, containing all of the pixels and channels
    plt.axis("off")
plt.waitforbuttonpress(-1)

#normalize the data(transform 0-255 values to 0-1)
X_train = X_train.astype("float32")/255
X_test = X_test.astype("float32")/255

#transform the data to one hot encoding
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = keras.models.Sequential()

#add input layer
model.add(keras.layers.Input(shape=X_train.shape[1:]))

#Conv-1 layer(Conv1-1, Conv1-2, MaxPooling2D)
#--------------------------------------------
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(keras.layers.Conv2D(filters = 32, kernel_size=3, padding="same", activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))
#--------------------------------------------


#Conv-2 layer(Conv2-1, Conv2-2, MaxPooling2D)
#--------------------------------------------
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
#--------------------------------------------

#Conv-3 layer(Conv3-1, Conv3-2, MaxPooling2D)
#--------------------------------------------
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
#--------------------------------------------


#Flatten the layers
#--------------------------------------------
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))
#--------------------------------------------

model.summary()
model.compile(
    optimizer="rmsprop",
    loss = keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)
#train the model
history = model.fit(X_train,
                    y_train,
                    batch_size=TrainingConfig.BATCH_SIZE,
                    epochs = TrainingConfig.EPOCHS,
                    verbose=1,
                    validation_split=0.3)

#plot the results of training process
train_loss = history.history["loss"]
train_acc  = history.history["accuracy"]
valid_loss = history.history["val_loss"]
valid_acc  = history.history["val_accuracy"]    
plot_results([train_loss,valid_loss], ylabel = "Loss", ylim = [0.0,5.0],
        metric_name=["Training Loss", "Validation Loss"], color=["g","b"] )

plot_results(
    [train_acc, valid_acc],
    ylabel="Accuracy",
    ylim=[0.0, 1.0],
    metric_name=["Training Accuracy", "Validation Accuracy"],
    color=["g", "b"],)
#using the save() method, the model will be saved to the file system in the 'SavedModel' format.
model.save(current_working_directory+os.sep+"model_dropout.h5")
model.save(current_working_directory+os.sep+"model_dropout.keras")

reloaded_model_dropout = models.load_model(current_working_directory+os.sep+'model_dropout.h5')


test_loss, test_acc = reloaded_model_dropout.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc*100:.3f}")



def evaluate_model(dataset, model):
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    num_rows = 3
    num_cols = 6

    # Retrieve a number of images from the dataset.
    data_batch = dataset[0 : num_rows * num_cols]

    # Get predictions from model.
    predictions = model.predict(data_batch)

    plt.figure(figsize=(20, 8))
    num_matches = 0

    for idx in range(num_rows * num_cols):
        ax = plt.subplot(num_rows, num_cols, idx + 1)
        plt.axis("off")
        plt.imshow(data_batch[idx])

        pred_idx = tf.argmax(predictions[idx]).numpy()
        truth_idx = np.nonzero(y_test[idx])

        title = str(class_names[truth_idx[0][0]]) + " : " + str(class_names[pred_idx])
        title_obj = plt.title(title, fontdict={"fontsize": 13})

        if pred_idx == truth_idx:
            num_matches += 1
            plt.setp(title_obj, color="g")
        else:
            plt.setp(title_obj, color="r")

        acc = num_matches / (idx + 1)
    print("Prediction accuracy: ", int(100 * acc) / 100)
    
    return

evaluate_model(X_test, reloaded_model_dropout)
plt.waitforbuttonpress(-1)

# Generate predictions for the test dataset.
predictions = reloaded_model_dropout.predict(X_test)

# For each sample image in the test dataset, select the class label with the highest probability.
predicted_labels = [np.argmax(i) for i in predictions]

# Convert one-hot encoded labels to integers.
y_test_integer_labels = tf.argmax(y_test, axis=1)

# Generate a confusion matrix for the test dataset.
cm = tf.math.confusion_matrix(labels=y_test_integer_labels, predictions=predicted_labels)

# Plot the confusion matrix as a heatmap.
plt.figure(figsize=[12, 6])


sn.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 12})
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
plt.waitforbuttonpress(-1)