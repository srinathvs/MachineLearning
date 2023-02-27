import time
import random
import csv
import numpy as np
import tensorflow.keras.metrics
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import datasets


def smooth_labels(labels, factor=.1):
    labels *= random.uniform(.01, factor)
    labels += (factor / labels.shape[1])
    return labels


def random_jitter_image(image, def_size=10):
    image_jittery = tf.image.resize(image, [image.shape[0] + def_size, image.shape[1] + def_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    tf.image.random_crop(image_jittery, size=[image.shape[0], image.shape[1], 3])
    image_jittery = tf.image.random_flip_left_right(image_jittery)
    return image_jittery


def preprocess_train_with_label(image_path, labels, image_type='jpeg'):
    f = tf.io.read_file(image_path)

    if image_type == 'jpeg':
        image = tf.image.decode_jpeg(f, channels=3)
    elif image_type == 'png':
        image = tf.image.decode_png(f, channels=3)
    else:
        image = tf.image.decode_image(f, channels=3)

    image = tf.cast(image, tf.float32)

    # image = random_jitter_image(image)

    image = (image - 127.5) / 127.5

    return (image, labels)

