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
                 block_batch_sizes=[],
                 block_steps=[]):

        self.session_id = session_id

        self.config_path = os.path.join('config', self.session_id + '.json')

        # Session config already exists
        if os.path.exists(self.config_path):
            with open(self.config_path) as config_file:
                self.config = json.load(config_file)
        else:
            self.config = {'id': self.session_id,
                           'phase': 0,
                           'steps': 0,
                           'latent_size': latent_size,
                           'n_fmap': n_fmap,
                           'n_blocks': n_blocks,
                           'block_batch_sizes':block_batch_sizes,
                           'block_steps':block_steps}

        self.phase = self.config['phase']
        self.steps = self.config['steps']
        self.block_batch_sizes = self.config['block_batch_sizes']
        self.block_steps = self.config['block_steps']

        self.gan = ProgressiveGAN(latent_size=self.config['latent_size'],
                                  n_fmap=self.config['n_fmap'],
                                  n_blocks=self.config['n_blocks'],
                                  input_shape=self.config['input_shape'])

        if self.phase == 0 and self.steps == 0:
            self.gen_models = self.gan.build_gen()
            self.dis_models = self.gan.build_dis()
            self.comp_models = self.gan.build_composite()
        else:
            # TODO: load models
            pass

        self.scaled_data =

    def train_step(self):

        b = np.floor((self.phase + 1) / 2.0)  # Block index
        m = self.phase % 2  # Model version (0:straight-through, 1:fade-in)

        g_model = self.gen_models[b][m]
        d_model = self.dis_models[b][m]
        comp_model = self.comp_models[b][m]

        # Update alpha
        if m == 1:
            update_fadein([g_model, d_model, comp_model], i, n_steps)

        # Prepare samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

        # Train discriminator
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)

        # Train generator
        z_input = generate_latent_points(latent_dim, n_batch)
        y_real2 = np.ones((n_batch, 1))
        g_loss = comp_model.train_on_batch(z_input, y_real2)

        return d_loss1, d_loss2, g_loss

    def train(self):



    def train_old(self, g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch):

        g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]

        # Scale dataset for base level
        gen_shape = g_normal.output_shape
        scaled_data = scale_dataset(dataset, gen_shape[1:])
        print('Scaled Data', scaled_data.shape)

        # Train base model
        train_epochs_old(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], n_batch[0])
        summarize_performance('tuned', g_normal, latent_dim)

        # Train each model level
        for i in range(1, len(g_models)):
            # Get level models
            [g_normal, g_fadein] = g_models[i]
            [d_normal, d_fadein] = d_models[i]
            [gan_normal, gan_fadein] = gan_models[i]

            # Scale dataset for current level
            gen_shape = g_normal.output_shape
            scaled_data = scale_dataset(dataset, gen_shape[1:])
            print('Scaled Data', scaled_data.shape)

            # Train transition model for next level
            train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, e_fadein[i], n_batch[i], True)
            summarize_performance('faded', g_fadein, latent_dim)

            # Train straight-through model
            train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[i], n_batch[i])
            summarize_performance('tuned', g_normal, latent_dim)


def scale_dataset(images, new_shape):
    images_list = list()
    for image in images:
        new_image = np.resize(image, new_shape)
        images_list.append(new_image)
    return np.asarray(images_list)


def summarize_performance(status, g_model, latent_dim, n_samples=25):
    # Naming
    gen_shape = g_model.output_shape
    name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)

    # Generate images
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    X = (X - X.min()) / (X.max() - X.min())
    square = int(np.sqrt(n_samples))
    for i in range(n_samples):
        plt.subplot(square, square, 1 + i)
        plt.axis('off')
        plt.imshow(X[i])

    # Save images
    filename1 = 'plot_%s.png' % name
    plt.savefig(root_dir + 'results/' + filename1)
    plt.close()

    # Save generator model
    filename2 = 'model_%s' % name
    save_model(g_model, filename2, gen_shape[1], 'pg1')

    print('>Saved: %s and %s' % (filename1, filename2))





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
