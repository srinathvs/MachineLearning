from config import *

import tensorflow as tf

from CustomModelComponents.Normalizations import SpectralNorm
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from CustomModelComponents.Normalizations import SpectralNorm


def smooth_labels(labels, factor=.05):
    labels += random.uniform(.01, factor)
    print(labels.shape)
    return labels


def random_jitter_image(image, def_size=20):
    image_jittery = tf.image.resize(image, [image.shape[0] + def_size, image.shape[1] + def_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    tf.image.random_crop(image_jittery, size=[IMAGE_DIM[0], IMAGE_DIM[1], 3])
    image_jittery = tf.image.random_flip_left_right(image_jittery)
    return image_jittery


def preprocess_train_with_label_no_jitter(image_path, labels, image_type='jpeg'):
    f = tf.io.read_file(image_path)

    if image_type == 'jpeg':
        image = tf.image.decode_jpeg(f, channels=3)
    elif image_type == 'png':
        image = tf.image.decode_png(f, channels=3)
    else:
        image = tf.image.decode_image(f, channels=3)

    image = tf.image.resize(image, (IMAGE_DIM[0], IMAGE_DIM[1]))
    image = random_jitter_image(image)
    image = tf.image.resize(image, (IMAGE_DIM[0], IMAGE_DIM[1]))
    image = tf.cast(image, tf.float32)
    labels = tf.cast(smooth_labels(labels), tf.float32)

    image = (image - 127.5) / 127.5

    return (image, labels)


def preprocess_train_images(image_path, image_type="jpeg"):
    print(tf.squeeze(image_path))
    f = tf.io.read_file(tf.squeeze(image_path))

    if image_type == 'jpeg':
        image = tf.image.decode_jpeg(f, channels=3)
    elif image_type == 'png':
        image = tf.image.decode_png(f, channels=3)
    else:
        image = tf.image.decode_image(f, channels=3)

    image = tf.image.resize(image, (IMAGE_DIM[0], IMAGE_DIM[1]))
    # image = random_jitter_image(image)
    image = tf.image.resize(image, (IMAGE_DIM[0], IMAGE_DIM[1]))
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5
    return image


class ShowImage(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        n = 6
        k = 0
        if epoch % 5 == 0:
            out = self.model.generator(tf.random.normal(shape=(36, self.latent_dim)))
            plt.figure(figsize=(16, 16))
            for i in range(n):
                for j in range(n):
                    ax = plt.subplot(n, n, k + 1)
                    plt.imshow((out[k] + 1) / 2, )
                    plt.axis('off')
                    k += 1
            plt.savefig("generated/gen_images_epoch_{}.png".format(epoch + 1))
            plt.close()


def create_dataset_from_csv(csv_path, use_labels=False):
    files = []
    labels = []

    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, dialect='excel')
        for row in csv_reader:
            if not use_labels:
                if row:
                    files.append(row)
            if use_labels:
                files.append(row[0])
                row[1] = row[1].replace(' ', ',')
                label_array = [float(e) for e in row[1].strip("[] \n").split(",")]
                labels.append(label_array)
    # print(files)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    if use_labels:
        labels = tf.convert_to_tensor(smooth_labels(np.asarray(labels, dtype=np.float32)))
        dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.map(preprocess_train_images, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=65, seed=43, reshuffle_each_iteration=True)
        tf.keras.backend.clear_session()

    if use_labels:
        real_images, real_class_labels = next(iter(dataset))
        show_real(real_images, real_class_labels)
    else:
        real_images = next(iter(dataset))
        show_real(real_images)