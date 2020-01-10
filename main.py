from matplotlib import pyplot
import tensorflow as tf
import numpy as np
import time
import cv2
import os


dir_train = '/home/willian/PycharmProjects/vizentec/modeldata/train/'
dir_val = '/home/willian/PycharmProjects/vizentec/modeldata/valid/'
dir_test = '/home/willian/PycharmProjects/vizentec/modeldata/test/'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
        dir_train,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
)
valid_generator = train_datagen.flow_from_directory(
    directory=dir_val,
    target_size=(150, 150),
    class_mode='categorical'
)
