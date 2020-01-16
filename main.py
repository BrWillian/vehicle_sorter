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


dir_train = '/home/willian/PycharmProjects/vizentec/modeldata/train'
dir_val = '/home/willian/PycharmProjects/vizentec/modeldata/valid'
dir_test = '/home/willian/PycharmProjects/vizentec/modeldata/test'


model = Sequential([
    VGG19(input_shape=(150, 150, 3), include_top=False, weights='imagenet'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(6, activation='softmax')
])


train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    dir_train,
    target_size=(150, 150),
    class_mode='categorical',
    color_mode='rgb'
)

valid_generator = test_datagen.flow_from_directory(
    directory=dir_val,
    target_size=(150, 150),
    class_mode='categorical',
    color_mode='rgb'
)
test_generator = test_datagen.flow_from_directory(
    directory=dir_test,
    target_size=(150, 150),
    class_mode='categorical',
    color_mode='rgb'
)

for data_batch, label_batch in train_generator:
    print(data_batch.shape, label_batch.shape)
    break


model.summary()

callback = tf.keras.callbacks.ModelCheckpoint('model_1.hdf5', monitor='val_loss', save_best_only=True, verbose=1, mode='auto')

model.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=valid_generator,
                              validation_steps=50, callbacks=[callback])


model_save = load_model('model_1.hdf5')

loss, result = model_save.evaluate_generator(test_generator)
print(result, loss)



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy / epochs')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Train', 'Validadation'])
plt.show()
