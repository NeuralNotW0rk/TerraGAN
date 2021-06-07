
import os
import json

import numpy as np

from model import StyleGAN
from util import save_image

session = 'test8'
steps = 250000

config_path = os.path.join('config', session + '.json')
with open(config_path) as config_file:
    config = json.load(config_file)

gan = StyleGAN(latent_size=config['latent_size'],
               img_size=config['img_size'],
               n_map_layers=config['n_map_layers'],
               fmap_min=config['fmap_min'],
               fmap_max=config['fmap_max'],
               fmap_scale=config['fmap_scale'])

gen, _ = gan.build_gen()

gen.load_weights('models/' + session + '/gen_' + str(steps) + '.h5', by_name=True)

n_layers = gan.n_blocks * 2

z1 = gan.random_latent(1)
z2 = gan.random_latent(1)
id = str(z1[0][0]) + '_' + str(z2[0][0])

noise = gan.random_noise(1)
seed = [np.asarray([config['const_seed']])]

for i in range(n_layers + 1):
    z = [z1] * (n_layers - i) + [z2] * i
    img = gen.predict(z + noise + seed)
    print(np.amax(img), np.amin(img))
    save_image(img[0], id, i, 'style_grid')
    save_image(np.clip(img[0], 0.0, 1.0) / np.amax(img[0]), id + 'clipped', i, 'style_grid')
