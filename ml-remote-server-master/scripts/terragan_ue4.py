import tensorflow as tf
import \
    unreal_engine as ue  # for remote logging only, this is a proxy import to enable same functionality as local variants
from mlpluginapi import MLPluginAPI

from tile_generator import *


class TerraGANAPI(MLPluginAPI):

    def on_setup(self):
        self.tg = TileGenerator('pgf6', 2, overlap=2)
        ue.log('TileGenerator loaded')

    def on_json_input(self, json_input):

        ue.log('Received inputs')
        latents = np.asarray(json_input['latents']).reshape((9, self.tg.pgg.latent_size))
        tile_ids = np.asarray(json_input['tile_ids'])

        tile_out = self.tg.generate_tile(latents, tile_ids)
        tile_out = tile_out[:, :, 1].flatten()
        ue.log('Tile generated')

        return {'tile_out': list(tile_out)}

    def on_begin_training(self):
        pass


def get_api():
    return TerraGANAPI.get_instance()
