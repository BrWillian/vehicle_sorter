from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import time
import cv2
import os


dir_train = '/kaggle/input/iara-sixs/modeldata/train'
dir_val = '/kaggle/input/iara-sixs/modeldata/valid'
dir_test = '/kaggle/input/iara-sixs/modeldata/test'

IMG_SHAPE = (150,150, 3)
VGG19_MODEL = VGG19(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

VGG19_MODEL.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(6, activation='softmax')

model = Sequential([
    VGG19_MODEL,
    global_average_layer,
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(2048, activation='relu'),
    prediction_layer
])


train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    dir_train,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=100
)

valid_generator = test_datagen.flow_from_directory(
    directory=dir_val,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=100
)
test_generator = test_datagen.flow_from_directory(
    directory=dir_test,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=100
)

for data_batch, label_batch in train_generator:
    print(data_batch.shape, label_batch.shape)
    break


model.summary()

callback = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/bestmodel1.hdf5', monitor='val_loss', save_best_only=True, verbose=1, mode='auto')

Adam = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=Adam,
               loss='categorical_crossentropy',
               metrics=['accuracy'])


train_size = train_generator.n
valid_size = valid_generator.n
test_size = test_generator.n

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_size/100,
                              epochs=100,
                              validation_data=valid_generator,
                              validation_steps=valid_size/100, callbacks=[callback])


model_save = load_model('bestmodel1.hdf5')

loss, result = model_save.evaluate_generator(test_generator)
print(result, loss)



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy / epochs')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Train', 'Validadation'])
plt.show()
