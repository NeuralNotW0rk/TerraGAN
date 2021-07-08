from PIL import Image

import os
import json

from layer import *

root_dir = ''


def random_latents(latent_size, n):
    return np.random.normal(0.0, 1.0, size=[n, latent_size])


def save_weights(model, name, version, session):
    path = root_dir + 'models/' + session + '/'
    if not os.path.exists(path):
        os.mkdir(path)

    model.save_weights(path + '{}_{}.h5'.format(name, version))


def load_weights(model, name, version, session):

    path = root_dir + 'models/' + session + '/'
    model.load_weights(path + '{}_{}.h5'.format(name, version),
                       by_name=True)

    return model


def save_image(data, name, block, num, session):
    path = root_dir + 'results/' + session + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    path += 'block_' + str(block) + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    x = tf.keras.preprocessing.image.array_to_img(data)
    x.save(path + name + '_' + str(block) + '_' + str(num) + '.png')


class Config(dict):

    def __init__(self, path):
        super(Config, self).__init__()
        self.path = path

    def load(self):
        if os.path.exists(self.path):
            with open(self.path) as config_file:
                self.update(json.load(config_file))

    def save(self):
        with open(self.path, 'w') as config_file:
            config_file.write(json.dumps(self, indent=4))


