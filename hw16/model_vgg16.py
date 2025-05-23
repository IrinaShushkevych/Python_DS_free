import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.applications import VGG16
from keras.optimizers import Adam

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

image_size = (32, 32)
train_images = tf.image.resize(train_images[..., np.newaxis], image_size)
train_images = np.repeat(train_images, 3, axis=-1)
test_images = tf.image.resize(test_images[..., np.newaxis], image_size)
test_images = np.repeat(test_images, 3, axis=-1)
train_images = np.array(train_images) / 255.0
test_images = np.array(test_images) / 255.0

num_classes = 10

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

vgg_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(image_size[0], image_size[1], 3)
    )


vgg_base.trainable = False

model = Sequential([
    vgg_base,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(num_classes, activation='softmax'),
])

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 128
epochs = 40

history = model.fit(train_images, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(test_images, test_labels))

_, accuracy = model.evaluate(test_images, test_labels)
print(f'Точність на тестових даних: {accuracy * 100:.2f}%')

vgg_base.summary()

vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg_base.trainable = True
set_trainable = False
for layer in vgg_base.layers:
    if layer.name == "block5_conv1":
        set_trainable = True
    if layer.name == "block5_conv2":
        set_trainable = True
    if layer.name == "block5_conv3":
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


modified_model = Sequential([
   vgg_base,
   Flatten(),
   Dense(256, activation="relu"),
   Dense(10, activation="softmax"),
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(train_images, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(test_images, test_labels))

_, accuracy = model.evaluate(test_images, test_labels)
print(f'Точність на тестових даних: {accuracy * 100:.2f}%')

model.save('model_2.keras')

