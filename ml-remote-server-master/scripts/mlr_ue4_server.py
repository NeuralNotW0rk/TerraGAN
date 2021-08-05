import tensorflow as tf
import \
    unreal_engine as ue  # for remote logging only, this is a proxy import to enable same functionality as local variants
from mlpluginapi import MLPluginAPI

from tile_generator import *


class TerraGANAPI(MLPluginAPI):

    def on_setup(self):
        self.tg = TileGenerator('pgf6', 2, overlap=2)

    def on_json_input(self, json_input):

        ue.log(json_input)

        latents = np.asarray(json_input["latents"])
        tile_ids = json_input['tile_ids']

        tile_out = self.tg.generate_tile(latents, tile_ids)

        return {'tile': tile_out.tolist()}

    def on_begin_training(self):
        pass


def get_api():
    return TerraGANAPI.get_instance()
