import numpy as np
import matplotlib.pyplot as plt
import os
import json

from layer import *
from model import *

root_dir = ''


def load_real_samples(filename):

    data = np.load(filename)
    X = data['arr_0']
    X = X.astype('float32')
    X = (X - 127.5) / 127.5

    return X


def generate_real_samples(dataset, n_samples):

    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))

    return X, y


def generate_latent_points(latent_dim, n_samples):

    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)

    return x_input


def generate_fake_samples(generator, latent_dim, n_samples):

    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = -np.ones((n_samples, 1))

    return X, y


def update_fadein(models, step, n_steps):

    alpha = step / float(n_steps - 1)
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)


class TrainingSession(object):

    def __init__(self,
                 session_id,
                 latent_size=100,
                 n_fmap=256,
                 n_blocks=6,
                 block_batch_sizes=(),
                 block_steps=(),
                 data_path='data/vfp_256.npz'):

        self.session_id = session_id

        self.config_path = os.path.join('config', self.session_id + '.json')

        # Session config already exists
        if os.path.exists(self.config_path):
            with open(self.config_path) as config_file:
                self.config = json.load(config_file)
        else:
            self.config = {'id': self.session_id,
                           'block': 0,
                           'steps': 0,
                           'model_version': 0,
                           'latent_size': latent_size,
                           'n_fmap': n_fmap,
                           'n_blocks': n_blocks,
                           'block_batch_sizes': list(block_batch_sizes),
                           'block_steps': list(block_steps),
                           'data_path': data_path}

        self.block = self.config['block']
        self.steps = self.config['steps']
        self.model_version = self.config['model_version']
        self.block_batch_sizes = self.config['block_batch_sizes']
        self.block_steps = self.config['block_steps']
        self.data_path = self.config['data_path']

        self.gan = ProgressiveGAN(latent_size=self.config['latent_size'],
                                  n_fmap=self.config['n_fmap'],
                                  n_blocks=self.config['n_blocks'],
                                  input_shape=self.config['input_shape'])

        if self.block == 0 and self.steps == 0:
            self.gen_models = self.gan.build_gen()
            self.dis_models = self.gan.build_dis()
            self.comp_models = self.gan.build_composite()
        else:
            # TODO: load models
            pass

        self.dataset = load_real_samples(data_path)
        self.scaled_dataset = None
        self.scale_dataset()

    def scale_dataset(self):

        images_list = list()
        res = 2 ** (2 + self.block)
        for image in self.dataset:
            new_image = np.resize(image, [res, res, 3])
            images_list.append(new_image)
        self.scaled_dataset = np.asarray(images_list)

    def next_block(self):

        self.block += 1
        self.model_version = 1
        self.scale_dataset()

    def train(self):

        if self.steps > self.block_steps[self.block]:
            if self.block == 0 or self.model_version == 0:
                self.next_block()
            elif self.block == self.gan.n_blocks - 1:
                pass
            else:
                self.model_version = 0

        g_model = self.gen_models[self.block][self.model_version]
        d_model = self.dis_models[self.block][self.model_version]
        comp_model = self.comp_models[self.block][self.model_version]

        batch_size = self.block_batch_sizes[self.block]

        # Update alpha if needed
        if self.model_version == 1:
            update_fadein([g_model, d_model, comp_model], self.steps, self.block_steps[self.block])

        # Prepare samples
        X_real, y_real = generate_real_samples(self.scaled_dataset, batch_size / 2)
        X_fake, y_fake = generate_fake_samples(g_model, self.gan.latent_size, batch_size / 2)

        # Train discriminator
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)

        # Train generator
        z_input = generate_latent_points(self.gan.latent_size, batch_size)
        y_real2 = np.ones((batch_size, 1))
        g_loss = comp_model.train_on_batch(z_input, y_real2)

        if self.steps % 100 == 0:
            print('\nStep ' + str(self.steps) + ':')
            print('D_real:', d_loss1)
            print('D_fake:', d_loss2)
            print('G:', g_loss)

        if self.steps % 500 == 0:
            self.save()

        if self.steps % 1000 == 0:
            self.evaluate()

    def save(self):
        # TODO: implement
        pass

    def evaluate(self):
        # TODO: implement
        pass

    def is_complete(self):

        return self.block == self.gan.n_blocks - 1 \
               and self.model_version == 0 \
               and self.steps > self.block_steps[self.block]


'''
if __name__ == '__main__':

    n_blocks = 6  # (2^2..2^(n+1))
    latent_dim = 100

    # Build individual models
    d_models = define_discriminator(n_blocks)
    g_models = define_generator(latent_dim, n_blocks)

    # Composite models
    gan_models = define_composite(d_models, g_models)

    # Load data
    dataset = load_real_samples('data/vfp_256.npz')
    print('Loaded', dataset.shape)

    # Train
    n_batch = [16, 16, 16, 8, 4, 4]
    n_epochs = [5, 8, 8, 10, 10, 10]
    train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_batch)
'''
