import os
import json
import numpy as np
import matplotlib.pyplot as plt
import util

from model import StyleGAN
import noise as gn

class TerrainGenerator:

    def __init__(self, session, segment_idx, steps=None):

        self.config_path = os.path.join('config', session + '.json')
        with open(self.config_path) as config_file:
            self.config = json.load(config_file)

        self.segment_idx = segment_idx

        self.gan = StyleGAN(latent_size=self.config['latent_size'],
                            img_size=self.config['img_size'],
                            n_map_layers=self.config['n_map_layers'],
                            fmap_min=self.config['fmap_min'],
                            fmap_max=self.config['fmap_max'],
                            fmap_scale=self.config['fmap_scale'])

        self.const_seed = self.config['const_seed']

        self.map = self.gan.build_map()
        self.gen_a, self.gen_b = self.gan.build_gen(segment_idx)

        if steps is None:
            self.steps = self.config['steps']
        else:
            self.steps = steps

        def weight_path(name): return 'models/' + session + '/' + name + '_' + str(self.steps) + '.h5'

        if self.config['n_map_layers'] > 0:
            self.map.load_weights(weight_path('map'))
        self.gen_a.load_weights(weight_path('gen'), by_name=True)
        self.gen_b.load_weights(weight_path('gen'), by_name=True)

        self.latent_field = None
        self.tiles_per_row = 0
        self.tile_res = 2 ** (self.segment_idx + 2)
        self.b_scaling = 2 ** (self.gan.n_blocks - self.segment_idx - 1)

        self.w_field = None
        self.latent_field = None
        self.style_field = None
        self.stride = None

        print('Initialization complete')

    def random_latent_field(self, field_res, cropping=0, overlap=0):
        print('Generating random latent field...')

        output_tile_res = self.tile_res - (2 * cropping)
        self.tiles_per_row = int((field_res - overlap) / (output_tile_res - overlap))

        seed = [np.asarray([self.const_seed])]

        self.w_field = np.zeros(shape=[self.tiles_per_row, self.tiles_per_row, self.gan.latent_size])
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
                z = self.gan.random_latent_list(1)
                w = []
                if self.config['n_map_layers'] > 0:
                    for i in range(len(z)):
                        w.append(self.map.predict(z[i]))
                else:
                    w = z

                self.w_field[i, j] = w[0]

                noise = self.gan.random_noise(1)

                # Crop inputs for generator A
                cutoff_idx = (self.segment_idx + 1) * 2
                w = w[:cutoff_idx]
                noise = noise[:cutoff_idx]

                # Generate intermediate latent tiles
                tile = self.gen_a.predict(w + noise + seed)[0]

                if cropping > 0:
                    tile = tile[cropping:-cropping, cropping:-cropping]

                tile *= weight_mask

                ia = i * (output_tile_res - overlap)
                ja = j * (output_tile_res - overlap)

                self.latent_field[ia:ia + output_tile_res, ja:ja + output_tile_res] += tile
                overlap_map[ia:ia + output_tile_res, ja:ja + output_tile_res] += weight_mask

        self.latent_field /= overlap_map

    def gradient_latent_field(self, field_res):

        self.tiles_per_row = int(field_res / self.tile_res)
        n_tiles = self.tiles_per_row ** 2

        z = np.zeros(shape=[self.tiles_per_row, self.tiles_per_row, self.gan.latent_size])
        for i in range(self.gan.latent_size):
            z[:, :, i] = gn.generate_perlin_noise_2d(shape=[self.tiles_per_row, self.tiles_per_row], res=[4, 4])

        w = []
        if self.config['n_map_layers'] > 0:
            for i in range(self.tiles_per_row):
                for j in range(self.tiles_per_row):
                    w.append(self.map.predict(np.asarray([z[i, j]]))[0])
        else:
            w = z

        noise = self.gan.random_noise(n_tiles)

        # Crop inputs for generator A
        cutoff_idx = (self.segment_idx + 1) * 2
        w = [np.asarray(w)] * cutoff_idx
        noise = noise[:cutoff_idx]

        seed = [np.asarray([self.const_seed] * n_tiles)]

        # Generate intermediate latent tiles
        tiles = self.gen_a.predict(w + noise + seed)

        rows = []

        for i in range(0, n_tiles, self.tiles_per_row):
            rows.append(np.concatenate(tiles[i:i + self.tiles_per_row], axis=1))

        self.latent_field = np.concatenate(rows, axis=0)

    def add_gradient_noise(self, factor=1, field_res=64):

        if self.latent_field is None:
            self.latent_field = np.zeros(shape=[field_res, field_res, self.gen_a.output[-1].shape[-1]])

        for i in range(self.latent_field.shape[-1]):
            self.latent_field[:, :, i] += gn.generate_fractal_noise_2d(shape=self.latent_field.shape[:-1], res=[4, 4], octaves=4, persistence=1) * factor
            # plt.imshow(self.latent_field[:, :, i])
            # plt.show()

    def build_style_field(self, stride, method='constant'):

        print('Building style field...')

        self.stride = stride
        styles_per_row = int((self.latent_field.shape[0] - self.tile_res) / self.stride + 1)
        z = np.zeros(shape=[styles_per_row, styles_per_row, self.gan.latent_size])
        style_1 = self.gan.random_latent(1)[0]

        if method == 'constant':
            pass
        if method == 'random':
            pass
        if method == 'gradient':
            style_2 = self.gan.random_latent(1)[0]
            for i in range(self.gan.latent_size):
                noise_res = 2 ** int(np.ceil(np.log2(styles_per_row)))
                g_noise = gn.generate_perlin_noise_2d(shape=[noise_res, noise_res], res=[1, 1])[:styles_per_row, :styles_per_row]
                g_noise = (g_noise + 1) / 2.0
                for x in range(styles_per_row):
                    for y in range(styles_per_row):
                        z[x, y, i] = style_1[i] * g_noise[x, y] + style_2[i] * (1 - g_noise[x, y])
        if method == 'inherit':
            for x in range(styles_per_row):
                for y in range(styles_per_row):
                    xs = 1.0 * x / styles_per_row * self.tiles_per_row
                    ys = 1.0 * y / styles_per_row * self.tiles_per_row

                    x1 = int(np.floor(xs))
                    y1 = int(np.floor(ys))

                    x2 = int(np.ceil(xs))
                    y2 = int(np.ceil(ys))

                    if x2 == self.tiles_per_row:
                        x2 = x1
                    if y2 == self.tiles_per_row:
                        y2 = y1

                    if x1 == x2:
                        wy1 = self.w_field[x1, y1]
                        wy2 = self.w_field[x1, y2]
                    else:
                        wy1 = (xs - x1) / (x2 - x1) * self.w_field[x1, y1]\
                            + (x2 - xs) / (x2 - x1) * self.w_field[x2, y1]
                        wy2 = (xs - x1) / (x2 - x1) * self.w_field[x1, y2]\
                            + (x2 - xs) / (x2 - x1) * self.w_field[x2, y2]

                    if y1 == y2:
                        w = wy1
                    else:
                        w = (ys - y1) / (y2 - y1) * wy1 + (y2 - ys) / (y2 - y1) * wy2

                    z[x, y] = w

        self.style_field = np.zeros(shape=[styles_per_row, styles_per_row, self.gan.latent_size])
        for i in range(styles_per_row):
            for j in range(styles_per_row):
                if self.config['n_map_layers'] > 0:
                    self.style_field[i, j] = self.map.predict(np.asarray([z[i, j]]))[0]
                else:
                    self.style_field[i, j] = z[i, j]

    def process_latent_field(self):

        print('Processing latent field...')

        # Variables
        output_res = self.latent_field.shape[0] * self.b_scaling
        output_tile_res = self.tile_res * self.b_scaling
        output_shape = [output_res, output_res, 3]

        weight_mask = np.zeros(shape=[output_tile_res, output_tile_res, 3])
        center = np.asarray([output_tile_res / 2, output_tile_res / 2])
        max_weight = np.linalg.norm(center)
        for i in range(output_tile_res):
            for j in range(output_tile_res):
                pixel = np.asarray([i, j])
                weight_mask[i, j, :] = (max_weight - np.linalg.norm(center - pixel)) ** 4 + 1

        # Initialize output
        output = np.zeros(shape=output_shape)
        overlap_map = np.zeros(shape=output_shape)

        # Move gen_b across latent field
        for i in range(self.style_field.shape[0]):
            for j in range(self.style_field.shape[1]):

                # Inputs
                w = np.asarray([self.style_field[i, j]])
                noise = tg.gan.random_noise(1)

                # Crop/broadcast inputs for generator B
                cutoff_idx = (self.segment_idx + 1) * 2
                w = [w] * (self.gan.n_blocks * 2 - cutoff_idx)
                noise = noise[cutoff_idx:]

                # Get tile from latent_field
                ia = i * self.stride
                ja = j * self.stride

                tile_a = self.latent_field[ia:ia + self.tile_res, ja:ja + self.tile_res]

                # Generate image from tile
                tile_b = self.gen_b.predict(w + noise + [np.asarray([tile_a])])[0]
                tile_b *= weight_mask

                # Add pixels on top of final output
                ib = ia * self.b_scaling
                jb = ja * self.b_scaling

                output[ib:ib + output_tile_res, jb:jb + output_tile_res] += tile_b
                overlap_map[ib:ib + output_tile_res, jb:jb + output_tile_res] += weight_mask

        # Divide pixels by number of overlaps to complete averaging
        print('Merging...')
        output /= overlap_map

        return output


if __name__ == '__main__':

    k = 2

    tg = TerrainGenerator('test9', segment_idx=k, steps=320000)

    tg.random_latent_field(field_res=2 ** (k + 5), overlap=2 ** (k + 1))

    tg.build_style_field(stride=2 ** (k + 0), method='inherit')

    out = tg.process_latent_field()

    out = np.clip(out, 0.0, 1.0)

    out = out - np.amin(out)

    out = out / np.amax(out)

    print(np.amin(out), np.amax(out))

    util.save_image(out, 'test_9_in', k, 'tiling_test')

