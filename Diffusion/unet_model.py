import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Reshape, Conv2DTranspose, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, MultiHeadAttention)
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa


class PositionalEmbeddings(tf.keras.layers.Layer):

    def __init__(self, dim):
        super().__init__()
        self.embedding_dim = dim

    def get_timestep_embedding(self, timesteps, embedding_dim: int):

        half_dim = self.embedding_dim // 2
        emb = tf.math.log(10000.) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.cast(timesteps, dtype = tf.float32)[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
        if embedding_dim % 2 == 1:
            emb = tf.pad(emb, [[0, 0], [0, 1]])
        return emb

    def call(self, time):
        return self.get_timestep_embedding(time, self.embedding_dim)


def res_block(x, filters, n_groups, temb):
    previous = x
    x = Conv2D(filters, 3, padding="same", )(x)  ### Convolution layer with padding same, so that the resolution remains the same

    x += Dense(filters)(tf.nn.silu(temb))[:, None, None, :]

    x = tf.nn.silu(tfa.layers.GroupNormalization(n_groups, axis=-1)(x))
    x = Conv2D(filters, 3, padding="same", )(x)

    # Project residual
    residual = Conv2D(filters, 1, padding="same", )(previous)
    x = tf.keras.layers.add([x, residual])  # Add back residual
    return x




def get_model(im_shape=(64, 64, 3), n_resnets=2, n_groups=8, attn_dim=32, n_heads=4, base_filters = 32, num_layers = 4):
    input_1 = Input(shape=im_shape)  ### image input
    input_2 = Input(shape=())  ### time input
    t_dim = im_shape[0] * 16

    # Entry block
    x = Conv2D(32, 3, padding="same")(input_1)
    temb = PositionalEmbeddings(t_dim)(input_2)
    temb = Dense(t_dim)(tf.nn.gelu(Dense(t_dim)(temb)))
    hs = [x]

    ### Downward Path
    for filters in [32, 64, 128, 256]:
        for _ in range(n_resnets):
            x = res_block(x, filters, n_groups, temb)  ### resblock

            if filters == 64:
                x = tfa.layers.GroupNormalization(groups=n_groups, axis=-1)(
                    MultiHeadAttention(num_heads=n_heads, key_dim=attn_dim, attention_axes=(1, 2), )(query=x, value=x))
        hs.append(x)  ### append the output features to hs
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)  ### Downsampling in order to move to the next resolution level

    ### Bottleneck
    x = res_block(x, 256, n_groups, temb)
    x = tfa.layers.GroupNormalization(groups=n_groups, axis=-1)(MultiHeadAttention(num_heads=n_heads, key_dim=attn_dim, attention_axes=(1, 2), )(query=x, value=x))
    x = res_block(x, 256, n_groups, temb)

    ### Upward path
    for filters in [256, 128, 64, 32]:

        x = tf.image.resize_with_pad(x, hs[-1].shape[1], hs[-1].shape[2])
        x = tf.concat([x, hs.pop()], axis=-1)

        for _ in range(n_resnets):
            x = res_block(x, filters, n_groups, temb)

            if filters == 64:
                x = tfa.layers.GroupNormalization(groups=n_groups, axis=-1)(MultiHeadAttention(num_heads=n_heads, key_dim=attn_dim, attention_axes=(1, 2), )(query=x, value=x))

        if filters != 32:
            x = Conv2DTranspose(filters, 3, strides=(2, 2), )(x)  ### Upsampling

    x = res_block(x, 32, n_groups, temb)
    outputs = Conv2D(3, 3, padding="same", )(x)

    # Define the model
    model = Model([input_1, input_2], outputs, name='unet')
    return model

