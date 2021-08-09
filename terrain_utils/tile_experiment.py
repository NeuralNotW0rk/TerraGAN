import os
import json
import numpy as np
import matplotlib.pyplot as plt
import util

from model import *
from util import *
from latent_manipulation import *
import noise as gn


class TerrainGenerator(Session):

    def __init__(self, session, segment_idx, steps=None):

        super(TerrainGenerator, self).__init__(session)

        self.segment_idx = segment_idx

        self.pgg = PGGAN(latent_size=self.config['latent_size'],
                         channels=self.config['channels'],
                         n_blocks=self.config['n_blocks'],
                         block_types=self.config['block_types'],
                         n_fmap=self.config['n_fmap'])

        self.gen_a, self.gen_b = self.pgg.build_gen_stable(segment_idx)

        if steps is None:
            self.steps = self.config['steps']
        else:
            self.steps = steps

        version = '{}_{}'.format(self.pgg.n_blocks - 1, self.steps)
        load_weights(self.gen_a, 'gen', version, self.session_id)
        load_weights(self.gen_b, 'gen', version, self.session_id)

        self.latent_field = None
        self.tiles_per_row = 0
        self.tile_res = self.pgg.interm_res
        self.b_scaling = self.pgg.final_res / self.tile_res

        print('Initialization complete')

    def random_latent_field(self, field_res, cropping=0, overlap=0, lm_version=None, lm_attribute=None,  alpha=1.0):
        print('Generating random latent field...')

        lm = LatentManipulator(self.session_id, lm_version)

        output_tile_res = self.tile_res - (2 * cropping)
        stride = output_tile_res - overlap

        self.tiles_per_row = int((field_res - output_tile_res + stride) / stride)

        self.latent_field = np.zeros(shape=[field_res, field_res, self.gen_a.output[-1].shape[-1]])
        overlap_map = self.latent_field.copy()

        weight_mask = np.zeros(shape=[output_tile_res, output_tile_res, self.gen_a.output[-1].shape[-1]])

        center = np.asarray([output_tile_res / 2, output_tile_res / 2])
        max_weight = np.linalg.norm(center)
        for i in range(output_tile_res):
            for j in range(output_tile_res):
                pixel = np.asarray([i, j])
                weight_mask[i, j, :] = (max_weight - np.linalg.norm(center - pixel)) ** 1 + 1

        latents = np.asarray(self.config['sample_latents'])
        print(latents.shape)
        latents = np.reshape(latents, newshape=[8, 8, 128])
        print(latents.shape)
        #latents = np.random.normal(0, 1, size=[self.tiles_per_row, self.tiles_per_row, self.pgg.latent_size])
        #latents_perlin = np.zeros(shape=[field_res, field_res, self.pgg.latent_size])
        #for i in range(self.pgg.latent_size):
        #    latents_perlin[:, :, i] = gn.generate_perlin_noise_2d(shape=[field_res, field_res], res=[4, 4])
        #latents_perlin = latents_perlin[:self.tiles_per_row, :self.tiles_per_row]
        #plt.imshow(latents_perlin[:, :, 0])
        #plt.show()
        #latents = alpha * latents + (1 - alpha) * latents_perlin

        for i in range(self.tiles_per_row):
            delta = (i / (self.tiles_per_row - 1) * 2 - 1) * -4.0
            for j in range(self.tiles_per_row):

                # Inputs
                # latent = random_latents(self.pgg.latent_size, 1)
                latent = latents[i, j]
                latent = lm.center_latent(latent, lm_attribute)
                latent = lm.move_latent(latent, lm_attribute, delta)

                # Generate intermediate latent tiles
                tile = self.gen_a.predict(np.asarray([latent]))[0]

                if cropping > 0:
                    tile = tile[cropping:-cropping, cropping:-cropping]

                tile *= weight_mask

                ia = i * (output_tile_res - overlap)
                ja = j * (output_tile_res - overlap)

                self.latent_field[ia:ia + output_tile_res, ja:ja + output_tile_res] += tile
                overlap_map[ia:ia + output_tile_res, ja:ja + output_tile_res] += weight_mask

        overlap_map += 1e-8
        self.latent_field /= overlap_map

    def add_gradient_noise(self, factor=1.0, field_res=64):

        if self.latent_field is None:
            self.latent_field = np.zeros(shape=[field_res, field_res, self.gen_a.output[-1].shape[-1]])

        for i in range(self.latent_field.shape[-1]):
            self.latent_field[:, :, i] += gn.generate_fractal_noise_2d(shape=self.latent_field.shape[:-1], res=[4, 4], octaves=4, persistence=1) * factor

    def process_latent_field(self, stride, blend=True):

        print('Processing latent field...')

        # Variables
        output_res = int(self.latent_field.shape[0] * self.b_scaling)
        output_tile_res = int(self.tile_res * self.b_scaling)
        output_shape = [output_res, output_res, self.pgg.channels]

        # Blending variables
        weight_mask = None
        overlap_map = None
        if blend:
            weight_mask = np.zeros(shape=[output_tile_res, output_tile_res, self.pgg.channels])
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
        steps = int((self.latent_field.shape[0] - self.tile_res + stride) / stride)
        for i in range(steps):
            delta = (i / (steps - 1) * 2 - 1) * -0.1
            for j in range(steps):

                # Get tile from latent_field
                ia = i * stride
                ja = j * stride

                tile_a = self.latent_field[ia:ia + self.tile_res, ja:ja + self.tile_res]

                # Generate image from tile
                tile_b = self.gen_b.predict(np.asarray([tile_a]))[0]
                #tile_b -= np.mean(tile_b[:, :, 1])
                #tile_b += delta

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

    tg = TerrainGenerator('pgf6', segment_idx=2)

    overlap = 4

    tg.random_latent_field(field_res=32,
                           overlap=overlap,
                           cropping=0,
                           lm_version='msm10',
                           lm_attribute='mean_5',
                           alpha=1)

    #tg.add_gradient_noise(factor=2)

    out = tg.process_latent_field(stride=4, blend=True)

    #out[:, :, 1] = combine_channels(out)[:, :, 0]
    #out[:, :, 0] = (out[:, :, 0] + 1.0) / 2.0

    out = (out + 1.0) / 2.0

    out = np.clip(out, 0.0, 1.0)

    print(np.amin(out), np.amax(out))

    util.save_image(out[:, :, 0:1], 'sample2_o2_0', 6, 1, 'tiling_test')
    util.save_image(out[:, :, 1:2], 'sample2_o2_1', 6, 1, 'tiling_test')

