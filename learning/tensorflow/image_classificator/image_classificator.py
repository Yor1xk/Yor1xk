import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
import random
import tensorflow as tf
from keras import datasets
from keras import layers
from keras import models
from keras import Sequential
from keras import utils
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

#convenience function for plotting the training results
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
    plt.xlim([0, 20])
    plt.ylim(ylim)
    # Tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)   
    plt.show()
    plt.waitforbuttonpress(-1)





#to_categorical() keras utils functions transforms an integer encoding of labels into a one - hot encoding

plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["image.cmap"] = "gray"


SEED_VALUE = 42

random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

mnist_dataset = datasets.mnist
#load data from the dataset
(X_train_all, Y_train_all), (X_test, Y_test) = mnist_dataset.load_data()

X_valid = X_train_all[:10000]
X_train = X_train_all[10000:]

y_valid = Y_train_all[:10000]
y_train = Y_train_all[10000:]

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

#show some examples
plt.figure(figsize=(18,5))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.axis(True)
    plt.imshow(X_train[i], cmap="gray")
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.waitforbuttonpress(-1)


#flatten the image input shape(28*28*1 becomes 784*1)
#normalize the values(to optimize computations we divide every value of the image (ranging from 0 to 255) so we get values ranging from 0 to 1)
X_train = X_train.reshape((X_train.shape[0], 28*28))
X_train = X_train.astype("float32")/255

X_test = X_test.reshape((X_test.shape[0], 28 * 28))
X_test = X_test.astype("float32") / 255

X_valid = X_valid.reshape((X_valid.shape[0], 28 * 28))
X_valid = X_valid.astype("float32") / 255

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

#transform integer labelling to one hot encoding
y_train = utils.to_categorical(y_train)
y_valid = utils.to_categorical(y_valid)
Y_test = utils.to_categorical(Y_test)

#instantiate the model
model = Sequential()

print(X_train.shape[0], X_train.shape[1])

#add input layer
model.add(layers.Input((X_train.shape[1],)))

#add hidden layers
model.add(layers.Dense(128, activation = "relu"))
model.add(layers.Dense(128, activation = "relu"))
model.add(layers.Dense(10, activation="softmax"))

#display model's summary
model.summary()

#compile the model
model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(X_train, y_train, epochs = 21, batch_size=64, validation_data=(X_valid,y_valid))

#retrieve training results.
train_loss = history.history["loss"]
train_acc  = history.history["accuracy"]
valid_loss = history.history["val_loss"]
valid_acc  = history.history["val_accuracy"]

#show the training and validation losses
plot_results([train_loss, valid_loss],ylabel = "Loss",
    ylim = [0.0, 0.5],
    metric_name= ["Training Loss", "Validation Loss"],
    color = ["g","b"]
    )

#show the training and validation accuracies
plot_results(
    [train_acc, valid_acc],
    ylabel="Accuracy",
    ylim=[0.9, 1.0],
    metric_name=["Training Accuracy", "Validation Accuracy"],
    color=["g", "b"],
)

#predict values on our test set
predictions = model.predict(X_test)

#for each image in predictions get the label representation 
predicted_values = [np.argmax(i) for i in predictions]

#get an integer encoding out of one hot encoding
y_test_labels = tf.argmax(Y_test, axis = 1)

#create a confusion matrix based on these infos
confusion_matrix = tf.math.confusion_matrix(labels=y_test_labels, predictions=predicted_values)

#show the confusion matrix with an annotated heatmap
plt.figure(figsize=(15,8))
sn.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 14})
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
plt.waitforbuttonpress(-1)















