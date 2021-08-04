

from util import *
from latent_manipulation import *


class TileGenerator(Session):

    def __init__(self, session_id, segment_idx, lm_version, steps=None):

        super(TileGenerator, self).__init__(session_id)

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

        self.tile_res = self.pgg.interm_res
        self.b_scaling = self.pgg.final_res / self.tile_res
