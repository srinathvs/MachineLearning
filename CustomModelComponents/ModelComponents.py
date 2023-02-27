import warnings

import tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

from CustomModelComponents.Normalizations import SpectralNorm, ConditionBatchNorm

warnings.filterwarnings('ignore')
print("Tensorflow", tf.__version__)


def hinge_loss_d(y_true, y_pred):
    return tf.keras.losses.Hinge()(y_pred, y_true)


def hinge_loss_g(y_true, y_pred):
    return -tf.reduce_mean(y_pred)


class SelfAttention(tensorflow.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # splitting input into batch size, height, width and channels of the image
        n, h, w, c = input_shape
        # finding the total pixel map size of the image as n_features
        self.n_features = h * w

        # Key, Query and Values are being computed
        self.conv_theta = Conv2D(c // 8, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_Theta')
        self.conv_phi = Conv2D(c // 8, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_Phi')
        self.conv_g = Conv2D(c // 2, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_G')

        self.conv_attention_g = Conv2D(c, 1, padding='same', kernel_constraint=SpectralNorm(), name='Conv_Attn_G')
        self.sigma = self.add_weight(shape=[1], initializer='zeros', trainable=True, name='Sigma')

    def call(self, input):
        n, h, w, c = input.shape

        theta = self.conv_theta(input)
        theta = tf.reshape(theta, (-1, self.n_features, theta.shape[-1]))

        phi = self.conv_phi(input)
        phi = tf.nn.max_pool2d(phi, ksize=2, strides=2, padding='VALID')
        phi = tf.reshape(phi, (-1, self.n_features // 4, theta.shape[-1]))

        g = self.conv_g(input)
        g = tf.nn.max_pool2d(g, ksize=2, strides=2, padding='VALID')
        g = tf.reshape(g, (-1, self.n_features // 4, g.shape[-1]))

        attention = tf.matmul(theta, phi, transpose_b=True)
        attention = tf.nn.softmax(attention)

        attention_g = tf.matmul(attention, g)
        attention_g = tf.reshape(attention_g, (-1, h, w, attention_g.shape[-1]))
        attention_g = self.conv_attention_g(attention_g)

        output = input + self.sigma * attention_g

        return output


class Resblock(tensorflow.keras.layers.Layer):
    def __init__(self, filters, n_class = 18, **kwargs):
        super(Resblock, self).__init__(**kwargs)
        self.filters = filters
        self.n_class = n_class

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = Conv2D(self.filters, 3, padding='same', name='conv2d_1',
                             kernel_constraint=SpectralNorm())
        self.conv_2 = Conv2D(self.filters, 3, padding='same', name='conv2d_2',
                             kernel_constraint=SpectralNorm())
        self.cbn_1 = ConditionBatchNorm(self.n_class)
        self.cbn_2 = ConditionBatchNorm(self.n_class)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.bn_3 = tf.keras.layers.BatchNormalization()

        self.learned_skip = False

        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = Conv2D(self.filters, 1, padding='same', name='conv2d_3',
                                 kernel_constraint=SpectralNorm())
            self.cbn_3 = ConditionBatchNorm(self.n_class)

    def call(self, input_tensor, labels):
        x = self.conv_1(input_tensor)
        # x = self.cbn_1(x, labels)
        x = self.bn1(x)
        x = tf.nn.leaky_relu(x, 0.2)

        x = self.conv_2(x)
        x = self.bn_2(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.leaky_relu(x, 0.2)

        if self.learned_skip:
            skip = self.conv_3(input_tensor)
            skip = self.bn_3(skip)
            # skip = tf.keras.layers.BatchNormalization()(skip)
            skip = tf.nn.leaky_relu(skip, 0.2)
        else:
            skip = input_tensor

        output = skip + x
        return output


class ResblockDown(tf.keras.layers.Layer):
    def __init__(self, filters, downsample=True, **kwargs):
        super(ResblockDown, self).__init__(**kwargs)
        self.filters = filters
        self.downsample = downsample

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = Conv2D(self.filters, 3, padding='same', kernel_constraint=SpectralNorm())
        self.conv_2 = Conv2D(self.filters, 3, padding='same', kernel_constraint=SpectralNorm())
        self.learned_skip = False

        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = Conv2D(self.filters, 1, padding='same', kernel_constraint=SpectralNorm())

    def down(self, x):
        return tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    def call(self, input_tensor):
        x = self.conv_1(input_tensor)
        x = tf.nn.leaky_relu(x, 0.2)

        x = self.conv_2(x)
        x = tf.nn.leaky_relu(x, 0.2)

        if self.downsample:
            x = self.down(x)

        if self.learned_skip:
            skip = self.conv_3(input_tensor)
            skip = tf.nn.leaky_relu(skip, 0.2)
            if self.downsample:
                skip = self.down(skip)
        else:
            skip = input_tensor
        output = skip + x
        return output