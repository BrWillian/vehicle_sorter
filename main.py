import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l1, l2
import numpy as np
import time
import cv2
import os

#Set Directory
dir_train = 'modeldata/train'
dir_val = 'modeldata/valid'

IMG_SHAPE = (300, 300, 3)
VGG19_MODEL = VGG19(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

#Traing weights VGG19
VGG19_MODEL.trainable = True
print(len(VGG19_MODEL.layers))

for layer in VGG19_MODEL.layers:
    print(layer, layer.trainable)

prediction_layer = tf.keras.layers.Dense(6, activation='softmax')

#Create model
model = Sequential([
    VGG19_MODEL,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, input_dim=128, kernel_regularizer=l2(0.01), activity_regularizer=l1(0.01)),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(4096, activation='relu'),
    prediction_layer
])

#Set DataGenerator
train_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    dir_train,
    target_size=(300, 300),
    class_mode='categorical',
    batch_size=32
)

valid_generator = test_datagen.flow_from_directory(
    directory=dir_val,
    target_size=(300, 300),
    class_mode='categorical',
    batch_size=32
)

#Print shape of datagenerator
for data_batch, label_batch in train_generator:
    print(data_batch.shape, label_batch.shape)
    break

#Create checkpoint and save best model
callback = tf.keras.callbacks.ModelCheckpoint('model.hdf5', save_best_only=True, monitor='val_accuracy',
                                              verbose=1, mode='auto')

#Configure Optimizers
Adam = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=Adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

train_size = train_generator.n
valid_size = valid_generator.n

#Train model
history = model.fit_generator(train_generator, epochs=50, validation_data=valid_generator, callbacks=[callback])