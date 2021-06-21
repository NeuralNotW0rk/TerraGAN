'''
Based on code found at https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/
'''

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import RandomNormal

from layer import *
from util import *

def layer_name(block_idx, layer_type, layer_idx):
    return 'block' + str(block_idx) + '_' + layer_type + str(layer_idx)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


class ProgressiveGAN(object):

    def __init__(self,
                 latent_size=100,
                 n_fmap=128,
                 n_blocks=6,
                 input_shape=(4, 4, 3)):

        self.latent_size = latent_size
        self.n_fmap = n_fmap
        self.n_blocks = n_blocks
        self.input_shape = input_shape

    def build_dis(self):

        # Initialize params
        init = RandomNormal(stddev=0.02)
        const = max_norm(1.0)

        model_list = list()

        # Base model
        in_image = Input(shape=self.input_shape, name=layer_name(0, 'input', 0))
        d = Conv2D(self.n_fmap, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const,
                   name=layer_name(0, 'conv', 0))(in_image)
        d = LeakyReLU(alpha=0.2, name=layer_name(0, 'lrelu', 0))(d)
        d = MinibatchStdev()(d)
        d = Conv2D(self.n_fmap, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                   name=layer_name(0, 'conv', 1))(d)
        d = LeakyReLU(alpha=0.2, name=layer_name(0, 'lrelu', 1))(d)
        d = Conv2D(self.n_fmap, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const,
                   name=layer_name(0, 'conv', 2))(d)
        d = LeakyReLU(alpha=0.2, name=layer_name(0, 'lrelu', 2))(d)
        d = Flatten()(d)
        out_class = Dense(1, name=layer_name(0, 'dense', 0))(d)
        model = Model(in_image, out_class)
        model.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        model_list.append([model, model])

        def block(block_idx, old_model, n_input_layers=3):

            # Initialize params
            init = RandomNormal(stddev=0.02)
            const = max_norm(1.0)
            in_shape = list(old_model.input.shape)
            input_shape = (in_shape[-2] * 2, in_shape[-2] * 2, in_shape[-1])

            # New input layer
            in_image = Input(shape=input_shape, name=layer_name(block_idx, 'input', 0))
            d = Conv2D(self.n_fmap, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const,
                       name=layer_name(block_idx, 'conv', 0))(in_image)
            d = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 0))(d)

            # New block
            d = Conv2D(self.n_fmap, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                       name=layer_name(block_idx, 'conv', 1))(d)
            d = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 1))(d)
            d = Conv2D(self.n_fmap, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                       name=layer_name(block_idx, 'conv', 2))(d)
            d = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 2))(d)
            d = AveragePooling2D()(d)
            block_new = d

            # Append new block to old model (skipping old input)
            for i in range(n_input_layers, len(old_model.layers)):
                d = old_model.layers[i](d)
            model1 = Model(in_image, d)
            model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

            # Downsample new input and connect to old model input
            downsample = AveragePooling2D()(in_image)
            block_old = old_model.layers[1](downsample)
            block_old = old_model.layers[2](block_old)

            # Combine using weighted sum into single straight-through model for fading
            d = WeightedSum()([block_old, block_new])
            for i in range(n_input_layers, len(old_model.layers)):
                d = old_model.layers[i](d)
            model2 = Model(in_image, d)
            model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

            # Return two models: one for the new resolution and one for fading in from the previous resolution
            return [model1, model2]

        # Build sub-models
        for i in range(1, self.n_blocks):
            old_model = model_list[i - 1][0]
            models = block(i, old_model)
            model_list.append(models)

        return model_list

    def build_gen(self):

        # Initialize parameters
        init = RandomNormal(stddev=0.02)
        const = max_norm(1.0)
        in_dim = 4

        model_list = list()

        # Base model
        in_latent = Input(shape=(self.latent_size,), name=layer_name(0, 'input', 0))
        g = Dense(self.n_fmap * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const,
                  name=layer_name(0, 'dense', 0))(in_latent)
        g = Reshape((in_dim, in_dim, 128))(g)
        g = Conv2D(self.n_fmap, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                   name=layer_name(0, 'conv', 0))(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2, name=layer_name(0, 'lrelu', 0))(g)
        g = Conv2D(self.n_fmap, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                   name=layer_name(0, 'conv', 1))(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2, name=layer_name(0, 'irelu', 1))(g)
        out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const,
                           name=layer_name(0, 'to_rgb', 0))(g)
        model = Model(in_latent, out_image)
        model_list.append([model, model])

        def block(block_idx, old_model):

            # Initialize params
            init = RandomNormal(stddev=0.02)
            const = max_norm(1.0)

            # Add new block to end of old model
            block_end = old_model.layers[-2].output
            upsampling = UpSampling2D()(block_end)
            g = Conv2D(self.n_fmap, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                       name=layer_name(block_idx, 'conv', 0))(upsampling)
            g = PixelNormalization()(g)
            g = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 0))(g)
            g = Conv2D(self.n_fmap, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                       name=layer_name(block_idx, 'conv', 1))(g)
            g = PixelNormalization()(g)
            g = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 1))(g)
            out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const,
                               name=layer_name(block_idx, 'to_rgb', 0))(g)
            model1 = Model(old_model.input, out_image)

            # Combine upsampled old output with new output via weighted sum
            out_old = old_model.layers[-1]
            out_image2 = out_old(upsampling)
            merged = WeightedSum()([out_image2, out_image])
            model2 = Model(old_model.input, merged)

            return [model1, model2]

        # Build sub-models
        for i in range(1, self.n_blocks):
            old_model = model_list[i - 1][0]
            models = block(i, old_model)
            model_list.append(models)

        return model_list


def build_composite( discriminators, generators):

    model_list = list()

    # Build composite models
    for i in range(len(discriminators)):
        g_models, d_models = generators[i], discriminators[i]

        # Straight-through model
        d_models[0].trainable = False
        model1 = Sequential()
        model1.add(g_models[0])
        model1.add(d_models[0])
        model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        # Stage transition model
        d_models[1].trainable = False
        model2 = Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        model_list.append([model1, model2])

    return model_list
