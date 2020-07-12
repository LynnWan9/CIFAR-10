import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.datasets import cifar10
from keras.utils import to_categorical
import pandas as pd
import numpy as np

# set seed
seed = 1
np.random.seed(seed)

# read data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train dimension
print(x_train.shape)
# x_test dimension
print(x_test.shape)

# normalize data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# one-hot-encode the output
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# compile network
model.compile(
    loss="categorical_crossentropy",
    optimizer='rmsprop',
    metrics=["accuracy"]
)
model.summary()

# fit the model
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    verbose=2
)

# test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test loss:', test_loss)
print('test accuracy:', test_acc)

# plot
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()

# print predictions
prediction = model.predict_classes(x_test)
print(prediction)
submission = pd.DataFrame({'id': range(1, 10001), 'class': prediction})
submission.to_csv("prediction.csv", index=False)
