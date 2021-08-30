import tensorflow as tf
import \
    unreal_engine as ue  # for remote logging only, this is a proxy import to enable same functionality as local variants
from mlpluginapi import MLPluginAPI

from tile_generator import *


class TerraGANAPI(MLPluginAPI):

    def on_setup(self):
        self.tg = TileGenerator('pgf6', 2, overlap=4)
        ue.log('TileGenerator loaded')

    def on_json_input(self, json_input):

        ue.log('Generating tile '
               + str(json_input['faces'][4]) + ' '
               + str(json_input['x'][4]) + ' '
               + str(json_input['y'][4]))
        tile_ids = np.asarray(json_input['tile_ids'])
        rotations = np.asarray(json_input['rotations'])
        latents = np.asarray(json_input['latents']).reshape((9, self.tg.pgg.latent_size))

        tile_out = self.tg.generate_tile(latents, tile_ids, rotations,
                                         str(json_input['faces'][4]) + ' '
                                         + str(json_input['x'][4]) + ' '
                                         + str(json_input['y'][4]))

        tile_out = tile_out[:, :, 1].flatten()
        ue.log('Tile generated')

        return {'tile_out': list(tile_out)}

    def on_begin_training(self):
        pass


def get_api():
    return TerraGANAPI.get_instance()
