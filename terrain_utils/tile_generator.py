from model import *
from util import *

import matplotlib.pyplot as plt
class TileGenerator(Session):

    def __init__(self, session_id, segment_idx, overlap=2, steps=None):

        super(TileGenerator, self).__init__(session_id)

        self.segment_idx = segment_idx
        self.overlap = overlap

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

        self.res_a = self.gen_a.outputs[0].shape[1]
        self.res_b = self.gen_b.outputs[0].shape[1]
        self.scale_b = self.res_b / self.res_a
        self.latent_tile_map = {}

        self.weight_mask_a = np.zeros(shape=(self.res_a, self.res_a, 1))
        r = (self.res_a - 1.0) / 2.0
        max_weight = np.linalg.norm(np.asarray([r, r]))
        for i in range(self.res_a):
            x = i - r
            for j in range(self.res_a):
                y = j - r
                weight = np.linalg.norm(np.asarray([x, y]))
                self.weight_mask_a[i, j, 0] = max_weight - weight + 1

        self.weight_mask_b = np.zeros(shape=(self.res_b, self.res_b, 1))
        r = (self.res_b - 1.0) / 2.0
        max_weight = np.linalg.norm(np.asarray([r, r]))
        for i in range(self.res_b):
            x = i - r
            for j in range(self.res_b):
                y = j - r
                weight = np.linalg.norm(np.asarray([x, y]))
                self.weight_mask_b[i, j, 0] = (max_weight - weight) ** 4 + 1

    def generate_tile(self, latents, tile_ids):

        chunk_size_a = self.res_a * 3 - self.overlap * 2

        chunk_a = np.zeros(shape=(chunk_size_a, chunk_size_a, self.gen_a.outputs[0].shape[-1]))
        weight_map_a = np.zeros(shape=(chunk_size_a, chunk_size_a, 1))

        tiles = []

        # Lookup intermediate tiles and generate if missing
        for i in range(9):
            tile = None
            tile_id = tile_ids[i]
            if tile_id != 'empty':
                try:
                    tile = self.latent_tile_map[tile_id]
                except KeyError:
                    tile = self.gen_a.predict(np.asarray([latents[i]]))[0]
                    self.latent_tile_map[tile_id] = tile

            tiles.append(tile)

        for i in range(3):
            y = (self.res_a - self.overlap) * i
            for j in range(3):
                x = (self.res_a - self.overlap) * j
                if tiles[i * 3 + j] is not None:
                    chunk_a[x:x + self.res_a, y:y + self.res_a] += tiles[i * 3 + j] * self.weight_mask_a
                    weight_map_a[x:x + self.res_a, y:y + self.res_a] += self.weight_mask_a

        chunk_a /= weight_map_a + 1e-8

        chunk_size_b = int(chunk_size_a * self.scale_b)

        chunk_b = np.zeros(shape=(chunk_size_b, chunk_size_b, self.gen_b.outputs[0].shape[-1]))
        weight_map_b = np.zeros(shape=(chunk_size_b, chunk_size_b, 1))

        for i in range(3):
            ya = (self.res_a - self.overlap) * i
            yb = int(ya * self.scale_b)
            for j in range(3):
                xa = (self.res_a - self.overlap) * j
                xb = int(xa * self.scale_b)

                tile_b = self.gen_b.predict(np.asarray([chunk_a[xa:xa + self.res_a, ya:ya + self.res_a]]))[0]
                tile_b = (tile_b + 1.0) / 2.0
                chunk_b[xb:xb + self.res_b, yb:yb + self.res_b] += tile_b * self.weight_mask_b
                weight_map_b[xb:xb + self.res_b, yb:yb + self.res_b] += self.weight_mask_b

        chunk_b /= weight_map_b + 1e-8

        tile_start = int((self.res_a - self.overlap) * self.scale_b)
        tile_out = chunk_b[tile_start:tile_start + self.res_b, tile_start:tile_start + self.res_b]

        return tile_out


if __name__ == '__main__':
    latents = random_latents(128, 9)
    tile_ids = ['0', '1', 'empty', '3', '4', '5', '6', '7', '8']

    tg = TileGenerator('pgf6', 2, 4)
    tile = tg.generate_tile(latents, tile_ids)
    plt.imshow(tile[:, :, 1])
    plt.show()
