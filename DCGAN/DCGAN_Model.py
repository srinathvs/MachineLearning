import math

from matplotlib import pyplot as plt

from ModelConfig import *

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Input, MaxPool2D, Conv2DTranspose, BatchNormalization, Flatten, Add, LeakyReLU
from tensorflow.keras.optimizers import Adam


class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name='d_loss')
        self.g_loss_metric = tf.keras.metrics.Mean(name='g_loss')

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def show_images(self, real_images, real_class_labels=None):
        if real_class_labels != None:
            num_images = 5
            images = real_images[:5]
            labels = real_class_labels[:5]
            print(labels)
            images = images * 0.5 + 0.5
            grid_row = 1
            grid_col = num_images
        else:
            num_images = 5
            images = real_images[:5]
            images = images * 0.5 + 0.5
            grid_row = 1
            grid_col = num_images

        scale = 2
        f, axarr = plt.subplots(grid_row, grid_col, figsize=(grid_col * scale, grid_row * scale))

        for row in range(grid_row):
            ax = axarr if grid_row == 1 else axarr[row]
            for col in range(grid_col):
                ax[col].imshow(images[row * grid_col + col])
                ax[col].axis('off')
            plt.show()

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        random_noise = tf.random.normal(shape=(batch_size, LATENT_DIM))
        fake_images = self.generator(random_noise)


        real_labels = tf.ones((batch_size, 1)) + 0.25 * tf.random.uniform((batch_size, 1), minval=-1, maxval=1)
        fake_labels = tf.zeros((batch_size, 1)) + 0.25 * tf.random.uniform((batch_size, 1), )

        with tf.GradientTape() as recorder:
            real_predictions = self.discriminator(real_images)
            d_loss_real = self.loss_fn(real_labels, real_predictions)

            fake_predictions = self.discriminator(fake_images)
            d_loss_fake = self.loss_fn(fake_labels, fake_predictions)

            d_loss = d_loss_real + d_loss_fake

        partial_derivatives = recorder.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(partial_derivatives, self.discriminator.trainable_weights))

        random_noise = tf.random.normal(shape=(batch_size, LATENT_DIM))
        flipped_fake_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as recorder:
            fake_predictions = self.discriminator(self.generator(random_noise))
            g_loss = self.loss_fn(flipped_fake_labels, fake_predictions)

        partial_derivatives = recorder.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(partial_derivatives, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {'d_loss': self.d_loss_metric.result(),
                'g_loss': self.g_loss_metric.result()}


class DCGAN:
    def __init__(self):
        self.outputImageSizeX = IMAGE_DIM[0]
        self.outputImageSizeY = IMAGE_DIM[1]
        self.latent_size = LATENT_DIM
        self.layer_size = int(math.log2(self.outputImageSizeX) - 2)

        self.base_filters = BASE_FILTERS

        self.generator = self.build_model_generator()
        self.generator.summary()

        self.discriminator = self.build_model_discriminator()
        self.discriminator.summary()

    def build_model_generator(self):
        input_layer = Input(shape=self.latent_size)

        x = Dense(4 * 4 * self.latent_size)(input_layer)
        x = tf.keras.layers.Reshape((4, 4, self.latent_size))(x)

        for layerIndex in range(self.layer_size):
            currentFilters = int(self.base_filters / math.pow(2, layerIndex))
            x = tf.keras.layers.Conv2DTranspose(currentFilters, kernel_size=4, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=.2)(x)

        output_layer = Conv2DTranspose(3, kernel_size=4, strides=1, activation=tf.keras.activations.tanh, padding='same')(x)

        model = tf.keras.models.Model(input_layer, output_layer)
        return model

    def build_model_discriminator(self):
        discriminator_filters = self.base_filters

        input_layer = Input(shape=(self.outputImageSizeX, self.outputImageSizeY, 3))

        x = Conv2D(discriminator_filters, kernel_size=4, strides=2, padding='same')(input_layer)
        x = LeakyReLU(alpha=.2)(x)

        for layerInx in range(self.layer_size - 2):
            current_filters = int(discriminator_filters * math.pow(2, layerInx))

            x = Conv2D(current_filters, kernel_size=4, strides=2, padding='same')(x)
            x = LeakyReLU(alpha=.2)(x)

        # x = Conv2D(1, kernel_size=4, strides=2, padding='same')(x)

        x = Flatten()(x)
        output_layer = Dense(1, 'sigmoid')(x)

        return tf.keras.models.Model(input_layer, output_layer)


if __name__ == '__main__':
    DCGANFactory = DCGAN()
    gan = GAN(DCGANFactory.discriminator, DCGANFactory.generator)
    gan.compile(
        d_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5),
        g_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5),
        loss_fn=tf.keras.losses.BinaryCrossentropy())
