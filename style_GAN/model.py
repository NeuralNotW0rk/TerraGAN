from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Activation, Dense, LeakyReLU, Flatten, AveragePooling2D,\
    Conv2D, Reshape, add
from math import log2

import tensorflow.keras.backend as K
import numpy as np
import random

from layers import Conv2DMod


def hinge_d(y_true, y_pred):
    return K.mean(K.relu(1.0 + (y_true * y_pred)))


def w_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def upsample(x):
    return K.resize_images(x, 2, 2, "channels_last", interpolation='bilinear')


def block_res(block_idx):
    return 2 ** (block_idx + 2)


def layer_name(block_idx, layer_type, idx):
    return 'block' + str(block_idx) + '_' + layer_type + str(idx)


class StyleGAN(object):

    def __init__(self,
                 latent_size=512,
                 img_size=512,
                 n_map_layers=8,
                 fmap_min=8,
                 fmap_max=512,
                 fmap_scale=8):

        # Variables
        self.latent_size = latent_size
        self.img_size = img_size
        self.n_map_layers = n_map_layers
        self.fmap_min = fmap_min
        self.fmap_max = fmap_max
        self.fmap_scale = fmap_scale

        self.n_blocks = int(log2(self.img_size) - 1)

    def block_filters(self, block_idx):
        return max(min((2 ** (self.n_blocks - block_idx - 1)) * self.fmap_scale, self.fmap_max), self.fmap_min)

    def build_dis(self):

        img_in = Input(shape=[self.img_size, self.img_size, 3], name='img_input')
        x = img_in

        def block(inp, block_idx):

            n_filters = self.block_filters(self.n_blocks - block_idx - 1)

            res = Conv2D(n_filters, 1, kernel_initializer='he_uniform')(inp)

            out = Conv2D(filters=n_filters, kernel_size=3, padding='same', kernel_initializer='he_uniform')(inp)
            out = LeakyReLU(0.2)(out)
            out = Conv2D(filters=n_filters, kernel_size=3, padding='same', kernel_initializer='he_uniform')(out)
            out = LeakyReLU(0.2)(out)

            out = add([res, out])

            if block_idx < self.n_blocks - 1:
                out = AveragePooling2D()(out)

            return out

        for i in range(self.n_blocks):
            x = block(x, i)

        x = Flatten()(x)

        x = Dense(1, kernel_initializer='he_uniform')(x)

        return Model(inputs=img_in, outputs=x, name='dis_base')

    def build_map(self):

        if self.n_map_layers <= 0:
          return None

        # Mapping network f
        in_z = Input(shape=[self.latent_size], name='z_input')
        w = in_z
        for i in range(self.n_map_layers):
            w = Dense(units=512, kernel_initializer='he_normal', name='map_fc' + str(i))(w)
            w = LeakyReLU(alpha=0.1, name='map_lrelu' + str(i))(w)

        return Model(inputs=in_z, outputs=w, name='map_f')

    def build_gen(self, block_segment=None):

        # Inputs
        style_in = []
        noise_in = []
        for i in range(self.n_blocks * 2):
            style_in.append(Input([self.latent_size], name='style_input' + str(i)))
            res = block_res(i // 2)
            noise_in.append(Input(shape=[res, res, 1], name='noise_input' + str(i)))

        # Blocks
        def block(inp, block_idx):

            n_filters = self.block_filters(block_idx)

            if block_idx > 0:
                # Custom upsampling because of clone_model issue
                out = Lambda(upsample, output_shape=[None, inp.shape[2] * 2, inp.shape[2] * 2, None],
                             name=layer_name(block_idx, 'upsample', 0))(inp)
            else:
                out = Activation('linear', name='block0_pass')(inp)

            layer_idx = block_idx * 2
            style = Dense(units=inp.shape[-1], kernel_initializer='he_uniform',
                          name=layer_name(block_idx, 'style', 0))(style_in[layer_idx])
            delta = Dense(units=n_filters, kernel_initializer='zeros',
                          name=layer_name(block_idx, 'delta', 0))(noise_in[layer_idx])

            out = Conv2DMod(filters=n_filters, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                            name=layer_name(block_idx, 'conv_mod', 0))([out, style])
            out = add([out, delta], name=layer_name(block_idx, 'add_noise', 0))
            out = LeakyReLU(0.2, name=layer_name(block_idx, 'lrelu', 0))(out)

            layer_idx += 1
            style = Dense(units=n_filters, kernel_initializer='he_uniform',
                          name=layer_name(block_idx, 'style', 1))(style_in[layer_idx])
            delta = Dense(units=n_filters, kernel_initializer='zeros',
                          name=layer_name(block_idx, 'delta', 1))(noise_in[layer_idx])

            out = Conv2DMod(filters=n_filters, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                            name=layer_name(block_idx, 'conv_mod', 1))([out, style])
            out = add([out, delta], name=layer_name(block_idx, 'add_noise', 1))
            out = LeakyReLU(0.2, name=layer_name(block_idx, 'lrelu', 1))(out)

            return out

        # Latent
        seed_in = Input(shape=[1], name='latent_seed')
        x = Dense(4 * 4 * 256, activation='relu', kernel_initializer='random_normal', name='latent_const')(seed_in)
        x = Reshape([4, 4, 256], name='latent_reshape')(x)

        # Blocks with a output and b input if segmented
        tile_out = None
        tile_in = None
        for i in range(self.n_blocks):
            x = block(x, i)
            if i == block_segment:
                tile_out = x
                tile_in = Input(shape=tile_out.shape[1:], name='tile_input')
                x = tile_in

        x = Conv2D(filters=3, kernel_size=1, padding='same', kernel_initializer='he_normal',
                   name='conv_final_rgb')(x)

        # Use values centered around 0, but normalize to [0, 1], providing better initialization
        x = Lambda(lambda y: y/2 + 0.5)(x)

        if block_segment is not None:
            layer_split = (block_segment + 1) * 2
            gen_a = Model(inputs=style_in[:layer_split] + noise_in[:layer_split] + [seed_in], outputs=tile_out,
                          name='gen_a')
            gen_b = Model(inputs=style_in[layer_split:] + noise_in[layer_split:] + [tile_in], outputs=x,
                          name='gen_b')
        else:
            gen_a = Model(inputs=style_in + noise_in + [seed_in], outputs=x, name='gen_base')
            gen_b = None

        return gen_a, gen_b

    def build_gen_eval(self, map, gen):

        # Generator Model for Evaluation

        latent_in = []
        style = []
        noise_in = []

        for i in range(self.n_blocks * 2):
            latent_in.append(Input([self.latent_size], name='latent_input' + str(i)))
            style.append(map(latent_in[-1]))
            res = block_res(i // 2)
            noise_in.append(Input(shape=[res, res, 1], name='eval_noise_input' + str(i)))

        seed_in = Input(shape=[1], name='eval_latent_seed')

        gen_base = gen(inputs=style + noise_in + [seed_in])

        gen_eval = Model(inputs=latent_in + noise_in + [seed_in], outputs=gen_base, name='gen_eval')

        return gen_eval

    def random_latent(self, n):
        return np.random.normal(0.0, 1.0, size=[n, self.latent_size]).astype('float32')

    def random_latent_list(self, n):
        n_layers = self.n_blocks * 2
        return [self.random_latent(n)] * n_layers

    def mixed_latent_list(self, n):
        n_layers = self.n_blocks * 2
        tt = int(random.random() * n_layers)
        p1 = [self.random_latent(n)] * tt
        p2 = [self.random_latent(n)] * (n_layers - tt)

        return p1 + [] + p2

    def random_noise(self, n):
        noise = []
        for i in range(self.n_blocks * 2):
            res = block_res(i // 2)
            noise.append(np.random.uniform(0.0, 1.0, size=[n, res, res, 1]).astype('float32'))
        return noise


if __name__ == '__main__':

    sgan = StyleGAN()
    ga, gb = sgan.build_gen(block_segment=None)
    ga.summary()
    if gb is not None:
        gb.summary()
    m = sgan.build_map()
    m.summary()
    ge = sgan.build_gen_eval(m, ga)
    ge.summary()


