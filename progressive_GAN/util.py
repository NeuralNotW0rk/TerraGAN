from PIL import Image
import numpy as np
import random
import os
from tensorflow.keras.models import model_from_json


def save_model(model, name, num, session):
    path = 'models/' + session + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    json = model.to_json()
    with open(path + name + ".json", "w") as json_file:
        json_file.write(json)

    model.save_weights(path + name + "_" + str(num) + ".h5")
