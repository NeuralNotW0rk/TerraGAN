import os
import json

import numpy as np
import tensorflow as tf

root_dir = '../terrain_utils/'


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


def save_image(data, name, block, num, session):
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    path = root_dir + 'results/' + session + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    path += 'block_' + str(block) + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    x = tf.keras.preprocessing.image.array_to_img(data * 255, scale=False)
    x.save(path + name + '_' + str(block) + '_' + str(num) + '.png')


def combine_channels(images):
    # Batch of images
    if len(images.shape) == 4:
        norms = (images[:, :, :, 0] + 1) / 2
        diffs = images[:, :, :, 1]
    else:
        norms = (images[:, :, 0:1] + 1) / 2
        diffs = images[:, :, 1:2]
    return norms + diffs


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


class Session(object):

    def __init__(self, session_id):

        self.session_id = session_id
        config_path = root_dir + 'config/' + self.session_id + '.json'
        self.config = Config(config_path)
        self.config.load()
