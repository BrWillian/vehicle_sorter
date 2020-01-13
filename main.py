from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
import cv2
import os


dir_train = '/home/willian/PycharmProjects/vizentec/modeldata/train'
dir_val = '/home/willian/PycharmProjects/vizentec/modeldata/valid'
dir_test = '/home/willian/PycharmProjects/vizentec/modeldata/test'

model = Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation='softmax')
])


train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    dir_train,
    target_size=(28, 28),
    batch_size=400,
    class_mode='categorical'
)

valid_generator = test_datagen.flow_from_directory(
    directory=dir_val,
    target_size=(28, 28),
    batch_size=400,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    directory=dir_test,
    target_size=(28, 28),
    batch_size=400,
    class_mode='categorical'
)

for data_batch, label_batch in train_generator:
    print(data_batch.shape, label_batch.shape)
    break


model.summary()

model.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

history = model.fit_generator(train_generator, epochs=30, validation_data=valid_generator)