import os
import json
import numpy as np
import matplotlib.pyplot as plt
import util

from model import *
from util import *
import noise as gn


class TerrainGenerator:

    def __init__(self, session, segment_idx, steps=None):

        self.config_path = os.path.join('config', session + '.json')
        with open(self.config_path) as config_file:
            self.config = json.load(config_file)

        self.segment_idx = segment_idx

        self.gan = ProgressiveGAN(latent_size=self.config['latent_size'],
                                  channels=self.config['channels'],
                                  n_blocks=self.config['n_blocks'],
                                  n_fmap=self.config['n_fmap'])

        self.gen_a, self.gen_b = self.gan.build_gen_split(segment_idx)

        if steps is None:
            self.steps = self.config['steps']
        else:
            self.steps = steps

        def weight_path(name): return 'models/' + session + '/' + name + '_' + str(self.gan.n_blocks)\
                                      + '_' + str(self.steps) + '.h5'

        load_weights(self.gen_a, 'gen', self.gan.n_blocks - 1, self.steps, session)
        load_weights(self.gen_b, 'gen', self.gan.n_blocks - 1, self.steps, session)

        self.latent_field = None
        self.tiles_per_row = 0
        self.tile_res = self.gan.interm_res
        self.b_scaling = self.gan.final_res / self.tile_res

        print('Initialization complete')

    def random_latent_field(self, field_res, cropping=0, overlap=0):
        print('Generating random latent field...')

        output_tile_res = self.tile_res - (2 * cropping)
        self.tiles_per_row = int((field_res - overlap) / (output_tile_res - overlap))

        self.latent_field = np.zeros(shape=[field_res, field_res, self.gen_a.output[-1].shape[-1]])
        overlap_map = self.latent_field.copy()

        weight_mask = np.zeros(shape=[output_tile_res, output_tile_res, self.gen_a.output[-1].shape[-1]])

        center = np.asarray([output_tile_res / 2, output_tile_res / 2])
        max_weight = np.linalg.norm(center)
        for i in range(output_tile_res):
            for j in range(output_tile_res):
                pixel = np.asarray([i, j])
                weight_mask[i, j, :] = (max_weight - np.linalg.norm(center - pixel)) ** 4 + 1

        for i in range(self.tiles_per_row):
            for j in range(self.tiles_per_row):

                # Inputs
                latent = random_latents(self.gan.latent_size, 1)

                # Generate intermediate latent tiles
                tile = self.gen_a.predict(latent)[0]

                if cropping > 0:
                    tile = tile[cropping:-cropping, cropping:-cropping]

                tile *= weight_mask

                ia = i * (output_tile_res - overlap)
                ja = j * (output_tile_res - overlap)

                self.latent_field[ia:ia + output_tile_res, ja:ja + output_tile_res] += tile
                overlap_map[ia:ia + output_tile_res, ja:ja + output_tile_res] += weight_mask

        self.latent_field /= overlap_map

    def add_gradient_noise(self, factor=1, field_res=64):

        if self.latent_field is None:
            self.latent_field = np.zeros(shape=[field_res, field_res, self.gen_a.output[-1].shape[-1]])

        for i in range(self.latent_field.shape[-1]):
            self.latent_field[:, :, i] += gn.generate_fractal_noise_2d(shape=self.latent_field.shape[:-1], res=[4, 4], octaves=4, persistence=1) * factor
            # plt.imshow(self.latent_field[:, :, i])
            # plt.show()

    def process_latent_field(self, stride, blend=True):

        print('Processing latent field...')

        # Variables
        output_res = int(self.latent_field.shape[0] * self.b_scaling)
        output_tile_res = int(self.tile_res * self.b_scaling)
        output_shape = [output_res, output_res, self.gan.channels]

        # Blending variables
        weight_mask = None
        overlap_map = None
        if blend:
            weight_mask = np.zeros(shape=[output_tile_res, output_tile_res, self.gan.channels])
            center = np.asarray([output_tile_res / 2, output_tile_res / 2])
            max_weight = np.linalg.norm(center)
            for i in range(output_tile_res):
                for j in range(output_tile_res):
                    pixel = np.asarray([i, j])
                    weight_mask[i, j, :] = (max_weight - np.linalg.norm(center - pixel)) ** 4 + 1
            overlap_map = np.zeros(shape=output_shape)

        # Initialize output
        output = np.zeros(shape=output_shape)

        # Move gen_b across latent field
        steps = int((self.latent_field.shape[0] - 1) / stride)
        for i in range(steps):
            for j in range(steps):

                # Get tile from latent_field
                ia = i * stride
                ja = j * stride

                tile_a = self.latent_field[ia:ia + self.tile_res, ja:ja + self.tile_res]

                # Generate image from tile
                tile_b = self.gen_b.predict(np.asarray([tile_a]))[0]

                # Add pixels on top of final output
                ib = int(ia * self.b_scaling)
                jb = int(ja * self.b_scaling)

                if blend:
                    tile_b *= weight_mask
                    output[ib:ib + output_tile_res, jb:jb + output_tile_res] += tile_b
                    overlap_map[ib:ib + output_tile_res, jb:jb + output_tile_res] += weight_mask
                else:
                    output[ib:ib + output_tile_res, jb:jb + output_tile_res] = tile_b

        if blend:
            output /= overlap_map

        return output


if __name__ == '__main__':

    tg = TerrainGenerator('pgf1', segment_idx=2)

    tg.random_latent_field(field_res=64, overlap=4)

    out = tg.process_latent_field(stride=4, blend=False)

    out = np.clip((out + 1.0) / 2.0, 0.0, 1.0)

    print(np.amin(out), np.amax(out))

    util.save_image(out, 'pg_1_replace', 6, 2, 'tiling_test')

