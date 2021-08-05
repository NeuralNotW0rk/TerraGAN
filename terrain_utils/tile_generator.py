from model import *
from util import *


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
        self.latent_tile_map = {}

        self.weight_mask = np.zeros(shape=(self.res_a, self.res_a, 1))
        r = (self.weight_mask.shape[0] - 1.0) / 2.0
        for i in range(self.res_a):
            x = i - r
            for j in range(self.res_a):
                y = j - r
                d = max(abs(x), abs(y))
                self.weight_mask[i, j, 0] = r - d

    def generate_tile(self, latents, tile_ids):

        latent_field_chunk = np.zeros(shape=self.gen_a.outputs[0].shape[1:])
        weight_map = np.zeros(shape=(self.res_a, self.res_a, 1))

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

        ov = self.overlap

        # Blend latent tiles
        if self.overlap > 0:

            # Top left
            if tiles[0] is not None:
                latent_field_chunk[:ov, -ov:] += tiles[0][-ov:, :ov] * self.weight_mask[-ov:, :ov]
                weight_map[:ov, -ov:] += self.weight_mask[-ov:, :ov]

            # Top middle
            if tiles[1] is not None:
                latent_field_chunk[:, -ov:] += tiles[1][:, :ov] * self.weight_mask[:, :ov]
                weight_map[:, -ov:] += self.weight_mask[:, :ov]

            # Top right
            if tiles[2] is not None:
                latent_field_chunk[-ov:, -ov:] += tiles[2][:ov, :ov] * self.weight_mask[:ov, :ov]
                weight_map[-ov:, -ov:] += self.weight_mask[:ov, :ov]

            # Left
            if tiles[3] is not None:
                latent_field_chunk[:ov, :] += tiles[3][-ov:, :] * self.weight_mask[-ov:, :]
                weight_map[:ov, :] += self.weight_mask[-ov:, :]

            # Middle
            if tiles[4] is not None:
                latent_field_chunk[:, :] += tiles[4][:, :] * self.weight_mask[:, :]
                weight_map[:, :] += self.weight_mask[:, :]

            # Right
            if tiles[5] is not None:
                latent_field_chunk[-ov:, :] += tiles[5][:ov, :] * self.weight_mask[:ov, :]
                weight_map[-ov:, :] += self.weight_mask[:ov, :]

            # Bottom left
            if tiles[6] is not None:
                latent_field_chunk[:ov, :ov] += tiles[6][-ov:, -ov:] * self.weight_mask[-ov:, -ov:]
                weight_map[:ov, :ov] += self.weight_mask[-ov:, -ov:]

            # Bottom middle
            if tiles[7] is not None:
                latent_field_chunk[:, :ov] += tiles[7][:, -ov:] * self.weight_mask[:, -ov:]
                weight_map[:, :ov] += self.weight_mask[:, -ov:]

            # Bottom right
            if tiles[8] is not None:
                latent_field_chunk[-ov:, :ov] += tiles[8][:ov, -ov:] * self.weight_mask[:ov, -ov:]
                weight_map[-ov:, :ov] += self.weight_mask[:ov, -ov:]

            latent_field_chunk /= weight_map + 1e-8

        # If no overlap, just use middle tile
        else:
            latent_field_chunk = tiles[4]

        tile_out = self.gen_b.predict(np.asarray([latent_field_chunk]))[0]

        return tile_out
