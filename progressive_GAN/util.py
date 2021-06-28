from PIL import Image
import numpy as np
import random
import os
from tensorflow.keras.models import model_from_json

from layer import *

root_dir = ''


def save_model(model, name, block, num, session):
    path = root_dir + 'models/' + session + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    path += 'block_' + str(block) + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    json = model.to_json()
    with open(path + name + '.json', 'w') as json_file:
        json_file.write(json)

    model.save_weights(path + name + "_" + str(num) + ".h5")


def save_model_list(model_list, name, n_blocks, block, num, session):

    for i in range(n_blocks):
        save_model(model_list[i][0], name + '_' + str(i) + '_0', block, num, session)
        save_model(model_list[i][1], name + '_' + str(i) + '_1', block, num, session)


def load_model(name, block, num, session):
    path = root_dir + 'models/' + session + '/block_' + str(block) + '/'
    file = open(path + name + ".json", 'r')
    json = file.read()
    file.close()

    mod = model_from_json(json, custom_objects={'PixelNormalization': PixelNormalization,
                                                'MinibatchStdev': MinibatchStdev,
                                                'WeightedSum': WeightedSum})
    mod.load_weights(path + name + "_" + str(num) + ".h5")

    return mod


def load_model_list(name, n_blocks, block, num, session):
    model_list = []
    for i in range(n_blocks):
        model_0 = load_model(name + '_' + str(i) + '_0', block, num, session)
        model_1 = load_model(name + '_' + str(i) + '_1', block, num, session)
        model_list.append([model_0, model_1])


def save_image(data, name, block, num, session):
    path = root_dir + 'results/' + session + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    path += 'block_' + str(block) + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    x = Image.fromarray(np.uint8(data * 255))
    x.save(path + name + '_' + str(num) + '.png')
