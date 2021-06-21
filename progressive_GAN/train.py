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
                 input_shape=(4, 4, 3),
                 block_batch_sizes=(),
                 block_steps=(),
                 data_path='data/vfp_256.npz'):

        self.session_id = session_id

        self.config_path = 'config/' + self.session_id + '.json'

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
                           'input_shape': input_shape,
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
            self.save()
        else:
            self.gen_models = load_model_list('gen', self.gan.n_blocks, self.block, self.steps, self.session_id)
            self.dis_models = load_model_list('dis', self.gan.n_blocks, self.block, self.steps, self.session_id)

        self.comp_models = build_composite(self.dis_models, self.gen_models)

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
        self.steps = 0

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
        X_real, y_real = generate_real_samples(self.scaled_dataset, int(batch_size / 2))
        X_fake, y_fake = generate_fake_samples(g_model, self.gan.latent_size, int(batch_size / 2))

        # Train discriminator
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)

        # Train generator
        z_input = generate_latent_points(self.gan.latent_size, batch_size)
        y_real2 = np.ones((batch_size, 1))
        g_loss = comp_model.train_on_batch(z_input, y_real2)

        if self.steps % 100 == 0:
            print('\nBlock ' + str(self.block)
                  + ', version ' + str(self.model_version)
                  + ', step ' + str(self.steps) + ':')
            print('D_real:', d_loss1)
            print('D_fake:', d_loss2)
            print('G:', g_loss)

        if self.steps % 500 == 0:
            self.save()

        if self.steps % 1000 == 0:
            self.evaluate(g_model)

        self.steps += 1

    def save(self):
        save_model_list(self.gen_models, 'gen', self.gan.n_blocks, self.block, self.steps, self.session_id)
        save_model_list(self.dis_models, 'dis', self.gan.n_blocks, self.block, self.steps, self.session_id)

        self.config['block'] = self.block
        self.config['steps'] = self.steps
        self.config['model_version'] = self.model_version
        with open(self.config_path, 'w') as config_file:
            config_file.write(json.dumps(self.config, indent=4))

        print('--Models saved')

    def evaluate(self, model):
        z = generate_latent_points(self.gan.latent_size, 64)
        imgs = model.predict(z)
        r = []

        for i in range(0, 64, 8):
            r.append(np.concatenate(imgs[i:i + 8], axis=1))

        c1 = np.concatenate(r, axis=0)
        c1 = np.clip(c1, 0.0, 1.0)

        save_image(c1, 'i_' + str(self.model_version), self.block, self.steps, self.session_id)
        print('--Images saved')

    def is_complete(self):

        return self.block == self.gan.n_blocks - 1 \
               and self.model_version == 0 \
               and self.steps > self.block_steps[self.block]


if __name__ == '__main__':

    ts = TrainingSession(session_id='pro_test',
                         latent_size=100,
                         n_fmap=128,
                         n_blocks=6,
                         block_batch_sizes=[16, 16, 16, 16, 16, 16],
                         block_steps=[10000, 10000, 10000, 10000, 10000, 10000],
                         data_path='data/vfp_256.npz')

    while not ts.is_complete():
        ts.train()
