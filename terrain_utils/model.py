from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU, Multiply, Add

from layer import *
from wrapper import *
from util import *


class PGGAN:

    def __init__(self,
                 latent_size=100,
                 channels=3,
                 n_blocks=7,
                 block_types=None,
                 init_res=4,
                 kernel_size=3,
                 kernel_initializer='he_normal',
                 padding='same',
                 output_activation='linear',
                 n_fmap=None):

        self.latent_size = latent_size
        self.channels = channels
        self.n_blocks = n_blocks
        print(block_types)
        if block_types is None:
            self.block_types = ['base'] + ['resize'] * (n_blocks - 1)
        else:
            self.block_types = block_types
        assert self.block_types[0] == 'base', 'first block must be type "base"'
        self.init_res = init_res
        self.final_res = init_res * (2 ** block_types.count('resize'))
        self.interm_res = None
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.output_activation = output_activation
        if isinstance(n_fmap, int):
            self.n_fmap = [n_fmap] * n_blocks
        else:
            assert len(n_fmap) == n_blocks, 'n_fmap must be int or list of ints size n_blocks'
            self.n_fmap = n_fmap

    def gen_resize_block(self, x, block):

        x = EqualizeLearningRate(Conv2D(self.n_fmap[block],
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        kernel_initializer=self.kernel_initializer),
                                 name='block{}_conv1'.format(block))(x)
        x = PixelNormalization()(x)
        x = LeakyReLU()(x)
        x = EqualizeLearningRate(Conv2D(self.n_fmap[block],
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        kernel_initializer=self.kernel_initializer),
                                 name='block{}_conv2'.format(block))(x)
        x = PixelNormalization()(x)
        x = LeakyReLU()(x)

        return x

    def gen_denorm_block(self, x, block):

        x = EqualizeLearningRate(Conv2D(self.n_fmap[block],
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        kernel_initializer=self.kernel_initializer),
                                 name='block{}_conv1'.format(block))(x)
        x = PixelNormalization()(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = EqualizeLearningRate(Dense(units=2), name='block{}_dense1'.format(block))(x)

        return x

    def build_gen(self):

        # Input block
        in_latent = Input(shape=(self.latent_size,),
                          name='input_latent')
        alpha = Input(shape=1, name='input_alpha')

        x = EqualizeLearningRate(Dense(units=self.n_fmap[0] * self.init_res * self.init_res,
                                       kernel_initializer=self.kernel_initializer),
                                 name='base_dense')(in_latent)
        x = PixelNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape((self.init_res, self.init_res, self.n_fmap[0]))(x)
        x = EqualizeLearningRate(Conv2D(self.n_fmap[0],
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        kernel_initializer=self.kernel_initializer),
                                 name='base_conv')(x)
        x = PixelNormalization()(x)
        x = LeakyReLU()(x)

        # Remaining blocks
        op = None
        for i in range(1, self.n_blocks - 1):
            if self.block_types[i] == 'resize':
                x = UpSampling2D()(x)
                x = self.gen_resize_block(x, i)
            else:
                print('Invalid block type at position {}: {}'.format(i, self.block_types[i]))

        # ----Base block output----
        if self.block_types[self.n_blocks - 1] == 'base':

            # Block output
            x = EqualizeLearningRate(Conv2D(self.channels,
                                            kernel_size=1,
                                            padding=self.padding,
                                            kernel_initializer=self.kernel_initializer,
                                            activation=self.output_activation),
                                     name='to_channels_{}'.format(self.n_blocks - 1))(x)

            model = Model(inputs=[in_latent, alpha], outputs=x, name='gen')

        # ----Resize block output----
        elif self.block_types[self.n_blocks - 1] == 'resize':

            # Final block
            x = UpSampling2D()(x)
            x1 = self.gen_resize_block(x, self.n_blocks - 1)

            # Get latest block output
            x1 = EqualizeLearningRate(Conv2D(self.channels,
                                             kernel_size=1,
                                             padding=self.padding,
                                             kernel_initializer=self.kernel_initializer,
                                             activation=self.output_activation),
                                      name='to_channels_{}'.format(self.n_blocks - 1))(x1)

            # Get previous block output
            x2 = EqualizeLearningRate(Conv2D(self.channels,
                                             kernel_size=1,
                                             padding=self.padding,
                                             kernel_initializer=self.kernel_initializer,
                                             activation=self.output_activation),
                                      name='to_channels_{}'.format(self.n_blocks - 2))(x)

            # Combine using weighted sum
            x1 = Multiply()([alpha, x1])
            x2 = Multiply()([1 - alpha, x2])
            x = Add()([x1, x2])

            model = Model(inputs=[in_latent, alpha], outputs=x, name='gen')

        # ----Denorm layer output----
        elif self.block_types[self.n_blocks - 1] == 'denorm':

            # Final block
            x1 = self.gen_denorm_block(x, self.n_blocks - 1)

            # Get previous block output
            x2 = EqualizeLearningRate(Conv2D(self.channels,
                                             kernel_size=1,
                                             padding=self.padding,
                                             kernel_initializer=self.kernel_initializer,
                                             activation=self.output_activation),
                                      name='to_channels_{}'.format(self.n_blocks - 2))(x)

            # Denormalize previous block output
            x1 = ImageDenormalization()([x2, x1])

            # Combine using weighted sum
            x1 = Multiply()([alpha, x1])
            x2 = Multiply()([1 - alpha, x2])
            x = Add()([x1, x2])

        return model

    def dis_block(self, x, block):

        x = EqualizeLearningRate(Conv2D(self.n_fmap[block],
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        kernel_initializer=self.kernel_initializer),
                                 name='block{}_conv1'.format(block))(x)
        x = LeakyReLU()(x)
        x = EqualizeLearningRate(Conv2D(self.n_fmap[block - 1],
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        kernel_initializer=self.kernel_initializer),
                                 name='block{}_conv2'.format(block))(x)
        x = LeakyReLU()(x)

        return x

    def build_dis(self):

        # Input
        in_image = Input(shape=[self.final_res, self.final_res, self.channels], name='input_image')
        alpha = Input(shape=1, name='input_alpha')
        x = in_image

        if self.n_blocks > 1:
            # Get input for latest block
            x1 = EqualizeLearningRate(Conv2D(self.n_fmap[self.n_blocks - 1],
                                             kernel_size=1,
                                             padding=self.padding,
                                             kernel_initializer=self.kernel_initializer),
                                      name='from_channels_{}'.format(self.n_blocks - 1))(x)
            x1 = LeakyReLU()(x1)
            x1 = self.dis_block(x1, self.n_blocks - 1)

            if self.block_types[self.n_blocks - 1] == 'resize':
                x1 = AveragePooling2D()(x1)
                x2 = AveragePooling2D()(x)
            elif self.block_types[self.n_blocks - 1] == 'denorm':
                x2 = ImageNormalization()(x)
            else:
                x2 = x

            # Get input for previous block
            x2 = EqualizeLearningRate(Conv2D(self.n_fmap[self.n_blocks - 2],
                                             kernel_size=1,
                                             padding=self.padding,
                                             kernel_initializer=self.kernel_initializer),
                                      name='from_channels_{}'.format(self.n_blocks - 2))(x2)
            x2 = LeakyReLU()(x2)

            # Combine using weighted sum
            x1 = Multiply()([alpha, x1])
            x2 = Multiply()([1 - alpha, x2])
            x = Add()([x1, x2])

            # Remaining blocks
            for i in range(self.n_blocks - 2, 0, -1):
                x = self.dis_block(x, i)
                if self.block_types[i] == 'resize':
                    x = AveragePooling2D()(x)

        else:
            x = EqualizeLearningRate(Conv2D(self.n_fmap[self.n_blocks - 1],
                                            kernel_size=1,
                                            padding=self.padding,
                                            kernel_initializer=self.kernel_initializer),
                                     name='from_channels_{}'.format(self.n_blocks - 1))(x)
            x = LeakyReLU()(x)

        # Output block
        x = MinibatchStdev()(x)
        x = EqualizeLearningRate(Conv2D(self.n_fmap[0],
                                        kernel_size=3,
                                        padding=self.padding,
                                        kernel_initializer=self.kernel_initializer),
                                 name='base_conv1')(x)
        x = LeakyReLU()(x)
        x = EqualizeLearningRate(Conv2D(self.n_fmap[0],
                                        kernel_size=4,
                                        padding='valid',
                                        kernel_initializer=self.kernel_initializer),
                                 name='base_conv2')(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = EqualizeLearningRate(Dense(units=1), name='base_dense')(x)

        model = Model(inputs=[in_image, alpha], outputs=x, name='dis')

        return model

    def build_gen_stable(self, split_idx=-1):

        # Input block
        in_latent = Input(shape=(self.latent_size,),
                          name='input_latent')

        x = EqualizeLearningRate(Dense(units=self.n_fmap[0] * self.init_res * self.init_res,
                                       kernel_initializer=self.kernel_initializer),
                                 name='base_dense')(in_latent)
        x = PixelNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape((self.init_res, self.init_res, self.n_fmap[0]))(x)
        x = EqualizeLearningRate(Conv2D(self.n_fmap[0],
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        kernel_initializer=self.kernel_initializer),
                                 name='base_conv')(x)
        x = PixelNormalization()(x)
        x = LeakyReLU()(x)

        out_tile = None
        in_tile = None

        # Remaining blocks
        for i in range(1, self.n_blocks):

            if i == split_idx:
                out_tile = x
                self.interm_res = out_tile.shape[1]
                in_tile = Input(shape=out_tile.shape[1:], name='tile_input')
                x = in_tile

            up = UpSampling2D()(x)
            x = self.gen_block(up, i)

        # Final block output
        x = EqualizeLearningRate(Conv2D(self.channels,
                                        kernel_size=1,
                                        padding=self.padding,
                                        kernel_initializer=self.kernel_initializer),
                                 name='to_channels_{}'.format(self.n_blocks - 1))(x)

        if split_idx > 0:
            gen_a = Model(inputs=in_latent, outputs=out_tile, name='gen_a')
            gen_b = Model(inputs=in_tile, outputs=x, name='gen_b')
            return gen_a, gen_b

        gen = Model(inputs=in_latent, outputs=x, name='gen')
        return gen

    def build_semantic_predictor(self, semantics):

        name = '_'.join(semantics)

        # Input
        in_image = Input(shape=[self.final_res, self.final_res, self.channels], name='input_image')

        x = EqualizeLearningRate(Conv2D(self.n_fmap[self.n_blocks - 1],
                                        kernel_size=1,
                                        padding=self.padding,
                                        kernel_initializer=self.kernel_initializer),
                                 name='from_channels_{}'.format(self.n_blocks - 1))(in_image)

        # Blocks
        for i in range(self.n_blocks - 1, 0, -1):
            x = self.dis_block(x, i)
            x = AveragePooling2D()(x)

        # Output block
        x = MinibatchStdev()(x)
        x = EqualizeLearningRate(Conv2D(self.n_fmap[0],
                                        kernel_size=3,
                                        padding=self.padding,
                                        kernel_initializer=self.kernel_initializer),
                                 name='base_conv1')(x)
        x = LeakyReLU()(x)
        x = EqualizeLearningRate(Conv2D(self.n_fmap[0],
                                        kernel_size=4,
                                        padding='valid',
                                        kernel_initializer=self.kernel_initializer),
                                 name='base_conv2')(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = EqualizeLearningRate(Dense(units=len(semantics)), name='{}_dense'.format(name))(x)

        model = Model(inputs=in_image, outputs=x, name='semantic_pred')

        return model
