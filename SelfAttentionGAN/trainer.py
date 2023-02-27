from CustomModelComponents.Normalizations import SpectralNorm
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from CustomModelComponents.Normalizations import SpectralNorm
from SelfAttentionGAN.model import SAGANFactory

from config import *


def smooth_labels(labels, factor=.1):
    labels *= random.uniform(.01, factor)
    labels += (factor / labels.shape[1])
    print(labels.shape)
    return labels


def random_jitter_image(image, def_size=10):
    image_jittery = tf.image.resize(image, [image.shape[0] + def_size, image.shape[1] + def_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    tf.image.random_crop(image_jittery, size=[image.shape[0], image.shape[1], 3])
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

    image = tf.image.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))

    image = tf.cast(image, tf.float32)
    labels = tf.cast(labels, tf.float32)

    image = (image - 127.5) / 127.5

    return (image, labels)


def create_dataset_from_csv(csv_path):
    files = []
    labels = []

    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, dialect='excel')
        for row in csv_reader:
            files.append(row[0])
            row[1] = row[1].replace(' ', ',')
            label_array = [float(e) for e in row[1].strip("[] \n").split(",")]
            labels.append(label_array)

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

    labels = tf.convert_to_tensor(smooth_labels(np.asarray(labels, dtype=np.float32)))
    dataset = tf.data.Dataset.from_tensor_slices((files, labels))

    dataset = dataset.map(preprocess_train_with_label_no_jitter, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True, deterministic=True)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=512, reshuffle_each_iteration=True)
    tf.keras.backend.clear_session()
    sagan = SAGANFactory(IMAGE_SHAPE, 18)
    sagan.train(iter(dataset), 50000)


if __name__ == '__main__':
    create_dataset_from_csv(pokemon_dataset_path)
