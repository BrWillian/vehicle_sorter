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

IMG_SHAPE = (128, 128, 3)
VGG19_MODEL = VGG19(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

# VGG19_MODEL.trainable = True
print(len(VGG19_MODEL.layers))

for layer in VGG19_MODEL.layers[:-4]:
    layer.trainable = False

for layer in VGG19_MODEL.layers:
    print(layer, layer.trainable)

prediction_layer = tf.keras.layers.Dense(6, activation='softmax')

model = Sequential([
    VGG19_MODEL,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(4096, activation='relu'),
    prediction_layer
])

train_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    dir_train,
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=100
)

valid_generator = test_datagen.flow_from_directory(
    directory=dir_val,
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=100
)
test_generator = test_datagen.flow_from_directory(
    directory=dir_test,
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=100
)

for data_batch, label_batch in train_generator:
    print(data_batch.shape, label_batch.shape)
    break

callback = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/bestmodel1.hdf5', monitor='val_loss',
                                              save_best_only=True, verbose=1, mode='auto')

Adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=Adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

train_size = train_generator.n
valid_size = valid_generator.n
test_size = test_generator.n

history = model.fit_generator(train_generator, steps_per_epoch=train_size / 100, epochs=200,
                              validation_data=valid_generator, validation_steps=valid_size / 100, callbacks=[callback])

model_save = load_model('/kaggle/working/bestmodel1.hdf5')

loss, result = model_save.evaluate_generator(test_generator)
print(result, loss)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy / epochs')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Train', 'Validadation'])
plt.show()