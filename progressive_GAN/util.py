from PIL import Image

import os

from layer import *

root_dir = ''


def save_weights(model, name, block, num, session):
    path = root_dir + 'models/' + session + '/'
    if not os.path.exists(path):
        os.mkdir(path)

    model.save_weights(path + '{}_{}_{}.h5'.format(name, block, num))


def load_weights(model, name, block, num, session):

    path = root_dir + 'models/' + session + '/'
    model.load_weights(path + '{}_{}_{}.h5'.format(name, block, num),
                       by_name=True)

    return model


def save_image(data, name, block, num, session):
    path = root_dir + 'results/' + session + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    path += 'block_' + str(block) + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    x = Image.fromarray(np.uint8(data * 255))
    x.save(path + name + '_' + str(block) + '_' + str(num) + '.png')
