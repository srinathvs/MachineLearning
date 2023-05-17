import random

from tensorflow import tanh
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
import math

from CustomModelComponents.Normalizations import SpectralNorm
from CustomModelComponents.DatasetSpecific.Pokemon import *
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import csv
from CustomModelComponents.Preprocessing import *
from CustomModelComponents.Normalizations import SpectralNorm
from CustomModelComponents.ModelComponents import SelfAttention, Resblock, ResblockDown
from config import *


def build_generator(z_dim, n_class, final_image_shape):
    DIM = 8

    num_upscaling = int(math.log2(final_image_shape[0]) - 3)
    z = layers.Input(shape=(z_dim))
    labels = layers.Input(shape=(n_class), dtype='int32')
    x = Dense(4 * 4 * DIM)(z)
    label_in = Dense(units=n_class)(labels)
    x = tf.keras.layers.Concatenate()([x, label_in])
    x = Dense(8 * 8 * DIM)(x)
    x = layers.Reshape((8, 8, DIM))(x)

    DIM_instance = DIM
    coarse_features = []
    for _ in range(num_upscaling):
        coarse_features.append(x)
        skip = Resblock(DIM_instance, n_class)(x, labels)
        x = Resblock(DIM_instance, n_class)(x, labels)
        skip2 = tf.keras.layers.Concatenate()([x, skip])
        skip2 = tf.keras.layers.UpSampling2D((2, 2))(skip2)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = Resblock(DIM_instance, n_class)(x, labels)
        x = tf.keras.layers.Concatenate()([x, skip2])

    coarse_layers = []
    for layerIns in coarse_features:
        num_upscaling = int(math.log2(final_image_shape[0]) - math.log2(layerIns.shape[1]))
        for number in range(num_upscaling):
            layerIns = tf.keras.layers.UpSampling2D((2, 2))(layerIns)
            layerIns = Resblock(1, n_class)(layerIns, labels)
        coarse_layers.append(layerIns)


    for layer in coarse_layers:
        x = tf.keras.layers.Concatenate()([x, layer])

    # x = layers.UpSampling2D((2, 2))(x)
    x = SelfAttention()(x)

    x = Resblock(DIM, n_class)(x, labels)
    output_image = tanh(Conv2D(3, 7, padding='same')(x))
    return Model([z, labels], output_image, name='generator')


def build_discriminator(n_class=18):
    DIM = 32
    input_image = Input(shape=IMAGE_SHAPE)
    # input_labels = Input(shape=1)

    # embedding = Embedding(n_class, 4 * DIM)(input_labels)

    # embedding = Flatten()(embedding)

    x = ResblockDown(DIM)(input_image)  # 64

    x = ResblockDown(2 * DIM)(x)  # 32

    x = ResblockDown(4 * DIM)(x)

    x = ResblockDown(8 * DIM)(x)

    x = SelfAttention()(x)

    x = ResblockDown(16 * DIM)(x)  # 16

    x = ResblockDown(32 * DIM)(x)  # 16

    x = ResblockDown(32 * DIM, False)(x)  # 4

    x = SelfAttention()(x)

    x = Flatten()(x)

    x1 = Dense(500, activation='sigmoid')(x)
    x1 = Dense(100, activation='sigmoid')(x1)
    x1 = Dense(80, activation='sigmoid')(x1)

    x2 = Dense(100, activation='tanh')(x)
    x2 = Dense(100, activation='tanh')(x2)
    x2 = Dense(50, activation='tanh')(x2)

    output = Dense(n_class, activation='sigmoid')(x1)
    output2 = Dense(1, activation='tanh')(x2)

    return Model(input_image, [output, output2], name='discriminator')


class SAGANFactory:
    def __init__(self, image_shape=IMAGE_SHAPE, n_class=18, z_dim=64):
        self.z_dim = z_dim
        self.n_class = n_class

        # Build models
        self.optimizer_d = Adam(1e-6)
        self.optimizer_g = Adam(1e-6)
        self.discriminator = build_discriminator(n_class)

        self.generator = build_generator(z_dim, n_class, IMAGE_SHAPE)

        self.generator.summary()
        self.discriminator.summary()

        # pred = self.discriminator(self.generator.output)
        # self.model = Model(self.generator.input, pred, name='model')
        # self.model.compile(optimizer=self.optimizer_g, loss=self.hinge_loss_g)

        self.hinge_loss = tf.keras.losses.Hinge()

    def hinge_loss_d(self, y, is_real):
        label = 1. if is_real else -1.
        loss = self.hinge_loss(y, label)
        return loss

    def mse_loss_d(self, y, labels):
        return tf.keras.losses.mean_squared_error(y, labels)

    def hinge_loss_g(self, y):
        return -tf.reduce_mean(y)

    def train_step(self, train_gen, step, showOnce=True):
        real_images, real_class_labels = next(train_gen)
        batch_size = real_class_labels.shape[0]
        print(batch_size)
        real_labels = 1
        fake_labels = 0

        z = tf.random.normal((batch_size, self.z_dim))

        label = np.zeros(shape=(batch_size, self.n_class))
        label_list = list(label)
        index = np.random.randint(0, self.n_class)
        index2 = np.random.randint(0, self.n_class)
        for label_instance in label_list:
            label_instance[index] = 1
            label_instance[index2] = 1

        fake_class_labels = smooth_labels(np.array(label_list, dtype=np.float32))
        # fake_class_labels = tf.random.uniform([batch_size], 0, self.n_class, tf.int32)
        # fake_class_labels = real_class_labels

        with tf.GradientTape() as d_tape, \
                tf.GradientTape() as g_tape:
            # forward pass
            fake_images = self.generator([z, fake_class_labels])
            pred_real_labels, pred_real_class = self.discriminator(real_images)
            pred_fake_labels, pred_fake_class = self.discriminator(fake_images)

            # discriminator losses
            loss_fake_labels = self.mse_loss_d(pred_fake_labels, fake_class_labels)
            loss_real_labels = self.mse_loss_d(pred_real_labels, real_class_labels)

            loss_real_class = self.hinge_loss_d(pred_real_class, True)
            loss_fake_class = self.hinge_loss_d(pred_fake_class, False)

            # total loss
            d_loss1 = tf.reduce_mean(tf.add(tf.cast(loss_fake_labels, tf.float32), tf.cast(loss_real_labels, tf.float32)))
            d_loss_hinge = .5 * (loss_real_class + loss_fake_class)
            d_loss_real = d_loss_hinge

            d_loss_real = d_loss_hinge + loss_real_labels
            print('\n')
            d_loss = d_loss1 + d_loss_hinge

            # d_loss = tf.reduce_mean(tf.cast(loss_real, tf.float32))
            d_gradients = d_tape.gradient(d_loss_real, self.discriminator.trainable_variables)
            self.optimizer_d.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

            # Generator Loss
            g_loss = self.hinge_loss_g(pred_fake_class)
            # g_loss = tf.reduce_mean(tf.cast(loss_fake, tf.float32))
            # g_loss = -tf.reduce_mean((tf.cast(loss_fake, tf.float32)))
            g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
            if step % 1 == 0:
                self.optimizer_g.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return g_loss, d_loss

    def show_val(self):
        images_per_class = 1
        num_images = 5
        z = tf.random.normal((images_per_class * num_images, self.z_dim))
        label = np.zeros(shape=(images_per_class * num_images, self.n_class))
        label_list = list(label)

        type_dict = recreate_type_dict()
        print(type_dict)
        for label_instance in label_list:
            index = np.random.randint(0, self.n_class)
            index2 = np.random.randint(0, self.n_class)
            label_instance[index] = 1
            label_instance[index2] = 1
            print(index, index2)
            type1 = [key for key in type_dict.keys() if type_dict[key] == str(index)]
            type2 = [key for key in type_dict.keys() if type_dict[key] == str(index2)]
            print('types are : ', type1, type2)

        labels = np.array(label_list, dtype=np.int32)
        print(labels[0])

        print()

        images = self.generator.predict([z, labels])
        images = images * 0.5 + 0.5
        grid_row = images_per_class
        grid_col = num_images

        scale = 2
        f, axarr = plt.subplots(grid_row, grid_col,
                                figsize=(grid_col * scale, grid_row * scale))

        for row in range(grid_row):
            ax = axarr if grid_row == 1 else axarr[row]
            for col in range(grid_col):
                ax[col].imshow(images[row * grid_col + col])
                ax[col].axis('off')
        plt.show()

    def train(self, train_gen, steps, interval=200):
        for i in range(steps):
            g_loss, d_loss = self.train_step(train_gen, i)
            msg = f'Step {i} g_loss {tf.reduce_sum(g_loss):.4f} d_loss {d_loss:.4f}'
            print(msg)
            if i % interval == 0:
                if i <= 4000 or i >= 10000:
                    if i > 5000:
                        interval = 1000
                    msg = f'Step {i} g_loss {tf.reduce_sum(g_loss):.4f} d_loss {d_loss:.4f}'
                    print(msg)
                    self.show_val()


if __name__ == '__main__':
    saganInstance = SAGANFactory()
