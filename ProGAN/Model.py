import os
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, LeakyReLU, Reshape, Input, Flatten, AveragePooling2D, Dense, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import time

BUFFER_SIZE = 200
IMAGE_RESOLUTION = 256
n_workers = tf.data.experimental.AUTOTUNE
BATCH_SIZE = {2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 8, 8: 4, 9: 4, 10: 4}
TRAIN_STEP_RATIO = {k: BATCH_SIZE[2] / v for k, v in BATCH_SIZE.items()}


class PixelNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def call(self, input_tensor):
        return input_tensor / tf.math.sqrt(tf.reduce_mean(input_tensor ** 2, axis=-1, keepdims=True) + self.epsilon)


class MinibatchStd(tf.keras.layers.Layer):
    def __init__(self, group_size=4, epsilon=1e-8):
        super(MinibatchStd, self).__init__()
        self.epsilon = epsilon
        self.group_size = group_size

    def call(self, input_tensor):
        n, h, w, c = input_tensor.shape
        x = tf.reshape(input_tensor, [self.group_size, -1, h, w, c])
        group_mean, group_var = tf.nn.moments(x, axes=(0), keepdims=False)
        group_std = tf.sqrt(group_var + self.epsilon)
        avg_std = tf.reduce_mean(group_std, axis=[1, 2, 3], keepdims=True)
        x = tf.tile(avg_std, [self.group_size, h, w, 1])

        return tf.concat([input_tensor, x], axis=-1)


class FadeIn(tf.keras.layers.Layer):
    @tf.function
    def call(self, input_alpha, a, b):
        alpha = tf.reduce_mean(input_alpha)
        y = alpha * a + (1. - alpha) * b
        return y


def wasserstein_loss(y_true, y_pred):
    return -tf.reduce_mean(y_true * y_pred)


# class Conv2D(tf.keras.layers.Layer):
#     def __init__(self, out_channels, kernel=3, gain=2, **kwargs):
#         super(Conv2D, self).__init__(kwargs)
#         self.kernel = kernel
#         self.out_channels = out_channels
#         self.gain = gain
#         self.pad = kernel != 1
#
#     def build(self, input_shape):
#         self.in_channels = input_shape[-1]
#
#         initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
#         self.w = self.add_weight(shape=[self.kernel,
#                                         self.kernel,
#                                         self.in_channels,
#                                         self.out_channels],
#                                  initializer=initializer,
#                                  trainable=True, name='kernel')
#
#         self.b = self.add_weight(shape=(self.out_channels,),
#                                  initializer='zeros',
#                                  trainable=True, name='bias')
#
#         fan_in = self.kernel * self.kernel * self.in_channels
#         self.scale = tf.sqrt(self.gain / fan_in)
#
#     def call(self, inputs):
#         if self.pad:
#             x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
#         else:
#             x = inputs
#         output = tf.nn.conv2d(x, self.scale * self.w, strides=1, padding="VALID") + self.b
#         return output
#
#
# class Dense(tf.keras.layers.Layer):
#     def __init__(self, units, gain=2., **kwargs):
#         super(Dense, self).__init__(kwargs)
#         self.units = units
#         self.gain = gain
#
#     def build(self, input_shape):
#         self.in_channels = input_shape[-1]
#         initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
#         self.w = self.add_weight(shape=[self.in_channels,
#                                         self.units],
#                                  initializer=initializer,
#                                  trainable=True, name='kernel')
#
#         self.b = self.add_weight(shape=(self.units,),
#                                  initializer='zeros',
#                                  trainable=True, name='bias')
#
#         fan_in = self.in_channels
#         self.scale = tf.sqrt(self.gain / fan_in)
#
#     # @tf.function
#     def call(self, inputs):
#         output = tf.matmul(inputs, self.scale * self.w) + self.b
#         return output

class ProgressiveGAN():
    def __init__(self, z_dim=512, resolution=512, start_log2_res=2):
        self.start_log2_res = start_log2_res
        self.resolution = resolution
        self.log2_resolution = int(np.log2(resolution))
        self.log2_res_to_filter_size = {
            0: 512,
            1: 512,
            2: 512,  # 4x4
            3: 512,  # 8x8
            4: 512,  # 16x16
            5: 512,  # 32x32
            6: 256,  # 64x64
            7: 128,  # 128x128
            8: 64,  # 256x256
            9: 32,  # 512x512
            10: 16}  # 1024x1024

        self.z_dim = z_dim
        self.initializer = tf.keras.initializers.RandomNormal(0., 1.)
        self.opt_init = {'learning_rate': 1e-3, 'beta_1': 0.0, 'beta_2': 0.99, 'epsilon': 1e-8}

        self.g_loss = 0.
        self.d_loss = 0.
        self.build_all_generators()
        self.build_all_discriminators()

        # initialize generator with the base
        dummy_alpha = tf.keras.layers.Input(shape=(1), name='DummyAlpha')
        rgb = self.to_rgb[2](self.generator_blocks[2].output)
        self.generator = Model([self.generator_blocks[2].input, dummy_alpha], rgb)

        # build base discriminator
        input_image = tf.keras.layers.Input(shape=(4, 4, 3))
        alpha = tf.keras.layers.Input(shape=(1))
        x = self.from_rgb[2](input_image)
        pred = self.discriminator_blocks[2](x)

        self.discriminator = Model([input_image, alpha], pred,
                                   name='discriminator_4x4')

        self.optimizer_discriminator = Adam(**self.opt_init)

        self.discriminator.trainable = False

        # build composite model
        pred = self.discriminator([self.generator.output, self.generator.input[1]])

        self.model = Model(inputs=self.generator.input,
                           outputs=pred)
        self.model.compile(loss=wasserstein_loss, optimizer=Adam(**self.opt_init))
        self.ckpt_path = f"./checkpoints/stylegan/"

    def build_all_generators(self):
        # build all the generator block
        self.to_rgb = {}
        self.generator_blocks = {}
        self.generator_blocks[2] = self.build_generator_base(self.z_dim)
        self.to_rgb[2] = self.build_to_rgb(4, self.log2_res_to_filter_size[2])

        for log2_res in range(3, self.log2_resolution + 1):
            res = 2 ** log2_res
            filter_n = self.log2_res_to_filter_size[log2_res]
            self.to_rgb[log2_res] = self.build_to_rgb(res, filter_n)

            input_shape = self.generator_blocks[log2_res - 1].output[0].shape
            gen_block = self.build_generator_block(log2_res, input_shape)
            self.generator_blocks[log2_res] = gen_block

    def build_generator_base(self, input_shape):

        input_tensor = tf.keras.layers.Input(shape=input_shape)
        x = PixelNorm()(input_tensor)
        x = Dense(8192)(x)
        x = Reshape((4, 4, 512))(x)
        x = LeakyReLU(0.2)(x)
        x = PixelNorm()(x)

        x = Conv2D(512, 3,
                   name='gen_4x4_conv1')(x)
        x = LeakyReLU(0.2)(x)
        x = PixelNorm()(x)

        return Model(input_tensor, x,
                     name='generator_base')

    def build_generator_block(self, log2_res, input_shape):
        res = 2 ** log2_res
        res_name = f'{res}x{res}'
        filter_n = self.log2_res_to_filter_size[log2_res]

        input_tensor = tf.keras.layers.Input(shape=input_shape)
        x = UpSampling2D((2, 2))(input_tensor)

        x = Conv2D(filter_n, 3, name=f'gen_{res_name}_conv1')(x)
        x = LeakyReLU(0.2)(x)
        x = PixelNorm()(x)

        x = Conv2D(filter_n, 3, name=f'gen_{res_name}_conv2')(x)
        x = LeakyReLU(0.2)(x)
        x = PixelNorm()(x)

        return Model(input_tensor, x,
                     name=f'genblock_{res}_x_{res}')

    def build_discriminator_block(self, log2_res, input_shape):

        filter_n = self.log2_res_to_filter_size[log2_res]
        input_tensor = tf.keras.layers.Input(shape=input_shape)

        # First conv
        x = Conv2D(filter_n, 3)(input_tensor)
        x = LeakyReLU(0.2)(x)

        # Second conv + downsample
        filter_n = self.log2_res_to_filter_size[log2_res - 1]
        x = Conv2D(filter_n, 3, )(x)
        x = LeakyReLU(0.2)(x)
        x = AveragePooling2D((2, 2))(x)

        res = 2 ** log2_res
        return Model(input_tensor, x,
                     name=f'disc_block_{res}_x_{res}')

    def build_discriminator_base(self, input_shape):

        input_tensor = tf.keras.layers.Input(shape=input_shape)

        x = MinibatchStd()(input_tensor)
        # x = input_tensor
        x = Conv2D(512, 3, name='gen_4x4_conv1')(x)
        x = LeakyReLU(0.2)(x)
        x = Flatten()(x)

        x = Dense(512, name='gen_4x4_dense1')(x)
        x = LeakyReLU(0.2)(x)

        x = Dense(1, name='gen_4x4_dense2')(x)

        return Model(input_tensor, x,
                     name='discriminator_base')

    def build_to_rgb(self, res, filter_n):

        return tf.keras.Sequential([tf.keras.layers.Input(shape=(res, res, filter_n)),
                                    Conv2D(3, 1, activation=None,
                                           name=f'to_rgb_{res}x{res}_conv')],
                                   name=f'to_rgb_{res}x{res}')

    def build_from_rgb(self, res, filter_n):

        return tf.keras.Sequential([tf.keras.layers.Input(shape=(res, res, 3)),
                                    Conv2D(filter_n, 1,name=f'from_rgb_{res}x{res}_conv'),
                                    LeakyReLU(0.2)],
                                    name=f'from_rgb_{res}x{res}')

    def build_all_discriminators(self):

        self.from_rgb = {}
        self.discriminator_blocks = {}

        # all but the final block
        for log2_res in range(self.log2_resolution, 1, -1):
            res = 2 ** log2_res
            filter_n = self.log2_res_to_filter_size[log2_res]
            self.from_rgb[log2_res] = self.build_from_rgb(res, filter_n)

            input_shape = (res, res, filter_n)
            self.discriminator_blocks[log2_res] = self.build_discriminator_block(log2_res, input_shape)

        # last block at 4x4 resolution
        log2_res = 2
        filter_n = self.log2_res_to_filter_size[log2_res]
        self.from_rgb[log2_res] = self.build_from_rgb(4, filter_n)
        res = 2 ** log2_res
        input_shape = (res, res, filter_n)
        self.discriminator_blocks[log2_res] = self.build_discriminator_base(input_shape)

    def grow_discriminator(self, log2_res):
        res = 2 ** log2_res

        input_image = tf.keras.layers.Input(shape=(res, res, 3))
        alpha = tf.keras.layers.Input(shape=(1))

        x = self.from_rgb[log2_res](input_image)
        x = self.discriminator_blocks[log2_res](x)

        downsized_image = AveragePooling2D((2, 2))(input_image)
        y = self.from_rgb[log2_res - 1](downsized_image)

        x = FadeIn()(alpha, x, y)
        for i in range(log2_res - 1, 1, -1):
            x = self.discriminator_blocks[i](x)

        self.discriminator = Model([input_image, alpha], x,
                                   name=f'discriminator_{res}_x_{res}')

        self.optimizer_discriminator = Adam(**self.opt_init)

    def grow_generator(self, log2_res):
        res = 2 ** log2_res
        alpha = tf.keras.layers.Input(shape=(1))

        x = self.generator_blocks[2].input

        for i in range(2, log2_res):
            x = self.generator_blocks[i](x)

        old_rgb = self.to_rgb[log2_res - 1](x)
        old_rgb = UpSampling2D((2, 2))(old_rgb)

        x = self.generator_blocks[log2_res](x)
        new_rgb = self.to_rgb[log2_res](x)
        rgb = FadeIn()(alpha, new_rgb, old_rgb)
        self.generator = Model([self.generator_blocks[2].input, alpha],
                               rgb,
                               name=f'generator_{res}_x_{res}')

    def grow_model(self, log2_res):
        tf.keras.backend.clear_session()
        res = 2 ** log2_res
        print(f"Growing model to {res}x{res}")

        self.grow_generator(log2_res)
        self.grow_discriminator(log2_res)

        self.discriminator.trainable = False

        latent_input = tf.keras.layers.Input(shape=(self.z_dim))
        alpha_input = tf.keras.layers.Input(shape=(1))
        fake_image = self.generator([latent_input, alpha_input])
        pred = self.discriminator([fake_image, alpha_input])

        self.model = Model(inputs=[latent_input, alpha_input],
                           outputs=pred)
        self.model.compile(loss=wasserstein_loss,
                           optimizer=Adam(**self.opt_init))

    def train_discriminator_wgan_gp(self, real_images, alpha):

        batch_size = real_images.shape[0]
        real_labels = tf.ones(batch_size)
        fake_labels = -tf.ones(batch_size)

        z = tf.random.normal((batch_size, self.z_dim))
        fake_images = self.generator([z, alpha])

        with tf.GradientTape() as gradient_tape, \
                tf.GradientTape() as total_tape:
            # forward pass
            pred_fake = self.discriminator([fake_images, alpha])
            pred_real = self.discriminator([real_images, alpha])

            epsilon = tf.random.uniform((batch_size, 1, 1, 1))
            interpolates = epsilon * real_images + (1 - epsilon) * fake_images
            gradient_tape.watch(interpolates)
            pred_fake_grad = self.discriminator([interpolates, alpha])

            # calculate losses
            loss_fake = wasserstein_loss(fake_labels, pred_fake)
            loss_real = wasserstein_loss(real_labels, pred_real)
            loss_fake_grad = wasserstein_loss(fake_labels, pred_fake_grad)

            # gradient penalty
            gradients_fake = gradient_tape.gradient(loss_fake_grad, [interpolates])
            gradient_penalty = self.gradient_loss(gradients_fake)

            # drift loss
            all_pred = tf.concat([pred_fake, pred_real], axis=0)
            drift_loss = 0.001 * tf.reduce_mean(all_pred ** 2)

            total_loss = loss_fake + loss_real + gradient_penalty + drift_loss

            # apply gradients
            gradients = total_tape.gradient(total_loss, self.discriminator.variables)

            self.optimizer_discriminator.apply_gradients(zip(gradients, self.discriminator.variables))

        return total_loss

    def train_step(self, log2_res, data_gen, alpha):
        real_images = next(data_gen)
        self.d_loss = self.train_discriminator_wgan_gp(real_images, alpha)

        real_images = next(data_gen)
        batch_size = real_images.shape[0]
        real_labels = tf.ones(batch_size)
        z = tf.random.normal((batch_size, self.z_dim))
        self.g_loss = self.model.train_on_batch([z, alpha], real_labels)

    def generate(self, z):
        images = self.generator([z, self.alpha])
        images = np.clip((images * 0.5 + 0.5) * 255, 0, 255)
        return images.astype(np.uint8)

    def checkpoint(self, z, log2_res, step):
        res = 2 ** log2_res
        prefix = f'res_{res}x{res}_{step}'

        path = os.path.join(self.ckpt_path, prefix)
        print('Saving checkpoint', path)

        for i in range(2, log2_res + 1):
            self.to_rgb[i].save(f'{path}/to_rgb_{i}')
            self.from_rgb[i].save(f'{path}/from_rgb_{i}')
            self.discriminator_blocks[i].save(f'{path}/d_{i}')
            self.generator_blocks[i].save(f'{path}/g_{i}')

        images = self.generate(z)
        # plot_images(images, log2_res, f"./save_images/{prefix}.jpg")

    def load_checkpoint(self, path):
        max_log2res = int(max([os.path.basename(d).split('_')[-1] for d in glob(path + '/d_*')]))
        for i in range(2, max_log2res + 1):
            self.to_rgb[i] = tf.keras.models.load_model(f'{path}/to_rgb_{i}')
            self.from_rgb[i] = tf.keras.models.load_model(f'{path}/from_rgb_{i}')
            self.discriminator_blocks[i] = tf.keras.models.load_model(f'{path}/d_{i}')
            self.generator_blocks[i] = tf.keras.models.load_model(f'{path}/g_{i}')

        self.grow_model(max_log2res)
        self.start_log2_res = max_log2res

    def gradient_loss(self, grad):
        loss = tf.square(grad)
        loss = tf.reduce_sum(loss, axis=np.arange(1, len(loss.shape)))
        loss = tf.sqrt(loss)
        loss = tf.reduce_mean(tf.square(loss - 1))
        self.penalty_const = 10
        loss = self.penalty_const * loss
        return loss

    def train(self, datasets, steps_per_phase=1000, tick_interval=500):
        self.val_z = tf.random.normal((12, self.z_dim))

        for log2_res in range(self.start_log2_res, self.log2_resolution + 1, 1):
            start_time = time.time()
            self.current_log2_res = log2_res

            res = 2 ** log2_res
            data_gen = iter(datasets[log2_res])
            print(f"Resolution {res}x{res}")

            for state in ['TRANSITION', 'STABLE']:
                if state == 'TRANSITION' and log2_res == 2:
                    continue

                steps = int(TRAIN_STEP_RATIO[log2_res] * steps_per_phase)
                interval = int(TRAIN_STEP_RATIO[log2_res] * tick_interval)
                for step in range(0, steps):
                    alpha = step / steps if state == 'TRANSITION' else 1.
                    self.alpha = np.array([[alpha]])

                    if step % interval == 0:
                        print('alpha', self.alpha)
                        elapsed = time.time() - start_time
                        start_time = time.time()
                        minutes = int(elapsed // 60)
                        seconds = int(elapsed % 60)
                        print(f"elapsed {minutes} min {seconds} sec")
                        msg = f"{state}. Resolution {res}x{res} Step {step}: g_loss {self.g_loss:.4f} d_loss {self.d_loss:.4f}"
                        print(msg)

                        self.checkpoint(self.val_z, log2_res, step)

                    self.train_step(log2_res, data_gen, self.alpha)

            if log2_res != self.log2_resolution:
                self.grow_model(log2_res + 1)


if __name__ == '__main__':
    gan = ProgressiveGAN(resolution=256)
