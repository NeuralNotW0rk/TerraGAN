from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU, Multiply, Add
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import RandomNormal

from layer import *
from util import *

kernel_initializer = 'he_normal'


class ProgressiveGan(object):

    def __init__(self,
                 latent_size=100,
                 channels=3,
                 n_blocks=6,
                 init_res=4,
                 kernel_size=3,
                 padding='same',
                 n_fmap=None):

        self.latent_size = latent_size
        self.channels = channels
        self.n_blocks = n_blocks
        self.init_res = init_res
        self.final_res = init_res * (2 ** (n_blocks - 1))
        self.kernel_size = kernel_size
        self.padding = padding
        if isinstance(n_fmap, int):
            self.n_fmap = [n_fmap] * n_blocks
        else:
            assert len(n_fmap) == n_blocks, 'n_fmap must be int or list of ints size n_blocks'
            self.n_fmap = n_fmap

    def gen_block(self, x, block):

        x = EqualizeLearningRate(Conv2D(self.n_fmap[block],
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        kernel_initializer=kernel_initializer),
                                 name='block{}_conv1'.format(block))(x)
        x = PixelNormalization()(x)
        x = LeakyReLU()(x)
        x = EqualizeLearningRate(Conv2D(self.n_fmap[block],
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        kernel_initializer=kernel_initializer),
                                 name='block{}_conv2'.format(block))(x)
        x = PixelNormalization()(x)
        x = LeakyReLU()(x)

        return x

    def build_gen(self, fade_in=True):

        # Input block
        in_latent = Input(shape=(self.latent_size,),
                          name='input_latent')
        x = EqualizeLearningRate(Dense(self.n_fmap[0] * self.init_res * self.init_res,
                                       kernel_initializer=kernel_initializer),
                                 name='base_dense')(in_latent)
        x = PixelNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape((self.init_res, self.init_res, self.n_fmap[0]))(x)
        x = EqualizeLearningRate(Conv2D(self.n_fmap[0],
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        kernel_initializer=kernel_initializer),
                                 name='base_conv')(x)
        x = PixelNormalization()(x)
        x = LeakyReLU()(x)

        # Remaining blocks
        up = None
        for i in range(1, self.n_blocks):
            up = UpSampling2D()(x)
            x = self.gen_block(up, i)

        # Final block output
        x = EqualizeLearningRate(Conv2D(self.channels,
                                        kernel_size=1,
                                        padding=self.padding,
                                        kernel_initializer=kernel_initializer),
                                 name='to_channels_{}'.format(self.n_blocks))(x)

        if fade_in:
            # Get latest block output
            x1 = x

            # Get previous block output
            x2 = EqualizeLearningRate(Conv2D(self.channels,
                                             kernel_size=1,
                                             padding=self.padding,
                                             kernel_initializer=kernel_initializer),
                                      name='to_channels_{}_prev'.format(self.n_blocks))(up)

            # Combine using weighted sum
            alpha = Input(shape=1, name='input_alpha')
            x1 = Multiply()([alpha, x1])
            x2 = Multiply()([1 - alpha, x2])
            x = Add()([x1, x2])

            # Fade-in model
            return Model(inputs=[in_latent, alpha], outputs=x, name='gen_fade')

        # Stable model
        return Model(inputs=in_latent, outputs=x, name='gen')

    def dis_block(self, x, block):

        x = EqualizeLearningRate(Conv2D(self.n_fmap[block],
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        kernel_initializer=kernel_initializer),
                                 name='block{}_conv1'.format(block))(x)
        x = LeakyReLU()(x)
        x = EqualizeLearningRate(Conv2D(self.n_fmap[block],
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        kernel_initializer=kernel_initializer),
                                 name='block{}_conv2'.format(block))(x)
        x = LeakyReLU()(x)

        return x

    def build_dis(self):

        # Input
        in_image = Input(shape=[self.final_res, self.final_res, self.channels], name='input_image')

        # Get input for latest block
        x1 = EqualizeLearningRate(Conv2D(self.n_fmap[self.n_blocks - 1],
                                         kernel_size=1,
                                         padding=self.padding,
                                         kernel_initializer=kernel_initializer),
                                  name='from_channels_{}'.format(self.n_blocks))(in_image)
        x1 = self.dis_block(x1, self.n_blocks - 1)
        x1 = AveragePooling2D()(x1)

        # Get input for previous block
        x2 = AveragePooling2D()(in_image)
        x2 = EqualizeLearningRate(Conv2D(self.n_fmap[self.n_blocks - 2],
                                         kernel_size=1,
                                         padding=self.padding,
                                         kernel_initializer=kernel_initializer),
                                  name='from_channels_{}_prev'.format(self.n_blocks))(x2)

        # Combine using weighted sum
        alpha = Input(shape=1, name='input_alpha')
        x1 = Multiply()([alpha, x1])
        x2 = Multiply()([1 - alpha, x2])
        x = Add()([x1, x2])

        # Remaining blocks
        for i in range(self.n_blocks - 2, 0, -1):
            x = self.dis_block(x, i)
            x = AveragePooling2D()(x)

        # Output block
        x = MinibatchStdev()(x)
        x = EqualizeLearningRate(Conv2D(self.n_fmap[0],
                                        kernel_size=3,
                                        padding=self.padding,
                                        kernel_initializer=kernel_initializer),
                                 name='base_conv1')(x)
        x = LeakyReLU()(x)
        x = EqualizeLearningRate(Conv2D(self.n_fmap[0],
                                        kernel_size=4,
                                        padding='valid',
                                        kernel_initializer=kernel_initializer),
                                 name='base_conv2')(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = EqualizeLearningRate(Dense(1), name='base_dense')(x)

        return Model(inputs=[in_image, alpha], outputs=x)


def layer_name(block_idx, layer_type, layer_idx):
    return 'block' + str(block_idx) + '_' + layer_type + str(layer_idx)


class ProgressiveGANArray(object):

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
            d = EqualizeLearningRate(Conv2D(self.n_fmap, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                       name=layer_name(block_idx, 'conv', 1)))(d)
            d = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 1))(d)
            d = EqualizeLearningRate(Conv2D(self.n_fmap, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                       name=layer_name(block_idx, 'conv', 2)))(d)
            d = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 2))(d)
            d = AveragePooling2D()(d)
            block_new = d

            # Append new block to old model (skipping old input)
            for i in range(n_input_layers, len(old_model.layers)):
                d = old_model.layers[i](d)
            model1 = Model(in_image, d)

            # Downsample new input and connect to old model input
            downsample = AveragePooling2D()(in_image)
            block_old = old_model.layers[1](downsample)
            block_old = old_model.layers[2](block_old)

            # Combine using weighted sum into single straight-through model for fading
            d = WeightedSum()([block_old, block_new])
            for i in range(n_input_layers, len(old_model.layers)):
                d = old_model.layers[i](d)
            model2 = Model(in_image, d)

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
        g = EqualizeLearningRate(Dense(self.n_fmap * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const,
                  name=layer_name(0, 'dense', 0)), name=layer_name(0, 'elr', 0))(in_latent)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2, name=layer_name(0, 'lrelu', 0))(g)
        g = Reshape((in_dim, in_dim, self.n_fmap))(g)
        g = EqualizeLearningRate(Conv2D(self.n_fmap, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                   name=layer_name(0, 'conv', 1)), name=layer_name(0, 'elr', 1))(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2, name=layer_name(0, 'lrelu', 1))(g)
        g = EqualizeLearningRate(Conv2D(self.n_fmap, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                   name=layer_name(0, 'conv', 2)), name=layer_name(0, 'elr', 2))(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2, name=layer_name(0, 'irelu', 2))(g)
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
            g = EqualizeLearningRate(Conv2D(self.n_fmap, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                       name=layer_name(block_idx, 'conv', 0)), name=layer_name(block_idx, 'elr', 0))(upsampling)
            g = PixelNormalization()(g)
            g = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 0))(g)
            g = EqualizeLearningRate(Conv2D(self.n_fmap, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                       name=layer_name(block_idx, 'conv', 1)), name=layer_name(block_idx, 'elr', 1))(g)
            g = PixelNormalization()(g)
            g = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 1))(g)
            out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const,
                               name=layer_name(block_idx, 'to_rgb', 0))(g)
            model1 = Model(old_model.input, out_image, name='gen_' + str(block_idx) + '_0')

            # Combine upsampled old output with new output via weighted sum
            out_old = old_model.layers[-1]
            out_image2 = out_old(upsampling)
            merged = WeightedSum()([out_image2, out_image])
            model2 = Model(old_model.input, merged, name='gen_' + str(block_idx) + '_1')

            return [model1, model2]

        # Build sub-models
        for i in range(1, self.n_blocks):
            old_model = model_list[i - 1][0]
            models = block(i, old_model)
            model_list.append(models)

        return model_list


def build_composite(discriminators, generators):

    model_list = list()

    # Build composite models
    for i in range(len(discriminators)):
        g_models, d_models = generators[i], discriminators[i]

        # Straight-through model
        d_models[0].trainable = False
        model1 = Sequential()
        model1.add(g_models[0])
        model1.add(d_models[0])

        # Stage transition model
        d_models[1].trainable = False
        model2 = Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])

        model_list.append([model1, model2])

    return model_list
