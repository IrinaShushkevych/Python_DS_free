import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.optimizers import Adam

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0
input_shape = (train_images.shape[1], train_images.shape[2], 1)
train_images = np.reshape(train_images, (-1, input_shape[0], input_shape[1], input_shape[2]))
test_images = np.reshape(test_images, (-1, input_shape[0], input_shape[1], input_shape[2]))

print(train_images.shape)
print(test_images.shape)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax'),
])

model.summary()

model.compile(
    optimizer = Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=40, batch_size=512)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

model.save('model_1.keras')