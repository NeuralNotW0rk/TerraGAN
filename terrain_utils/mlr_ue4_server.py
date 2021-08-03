import tensorflow as tf
import \
    unreal_engine as ue  # for remote logging only, this is a proxy import to enable same functionality as local variants
from mlpluginapi import MLPluginAPI

from latent_manipulation import *


class TerraGANAPI(MLPluginAPI):

    def on_setup(self):
        lm = LatentManipulator('pgf6', )
        pass

    def on_json_input(self, json_input):

        ue.log(json_input)

        feed_dict = {self.a: json_input['a'], self.b: json_input['b']}

        raw_result = self.sess.run(self.c, feed_dict)

        ue.log('raw result: ' + str(raw_result))

        return {'c': raw_result.tolist()}

    def on_begin_training(self):
        pass


def get_api():
    return TerraGANAPI.get_instance()
