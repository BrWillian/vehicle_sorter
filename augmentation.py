import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import shutil
import os


def generate_images(dir):

    datagen = ImageDataGenerator(
        horizontal_flip=True
    )

    try:
        os.makedirs('images_data_augmentation')
    except:
        shutil.rmtree('images_data_augmentation')
        os.makedirs('images_data_augmentation')

    for root, _, files in os.walk(dir):
        for f in files:
            img = load_img(root+f)
            x = img_to_array(img)  #transforma imagem em array
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir='images_data_augmentation', save_prefix='aug', save_format='jpg'):
                i += 1
                if i > 1:
                    break


generate_images('/home/willian/PycharmProjects/vizentec/modeldata/valid/ao/')