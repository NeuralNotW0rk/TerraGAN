from tensorflow.keras.models import model_from_json
from PIL import Image

import os
import random
import numpy as np


from layers import Conv2DMod


def save_model(model, name, num, session):
    path = 'models/' + session + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    json = model.to_json()
    with open(path + name + ".json", "w") as json_file:
        json_file.write(json)

    model.save_weights(path + name + "_" + str(num) + ".h5")


def load_model(name, num, session):
    path = 'models/' + session + '/'
    file = open(path + name + ".json", 'r')
    json = file.read()
    file.close()

    mod = model_from_json(json, custom_objects={'Conv2DMod': Conv2DMod})
    mod.load_weights(path + name + "_" + str(num) + ".h5")

    return mod


def save_image(data, name, num, session):
    path = 'results/' + session + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    x = Image.fromarray(np.uint8(data * 255))
    x.save(path + name + '_' + str(num) + '.png')
