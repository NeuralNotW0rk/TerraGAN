from tensorflow.keras.optimizers import Adam
from math import floor
from random import random

import time
import os
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from model import StyleGAN
from datagen import dataGenerator
import util

mixed_prob = 0.9
channels = 1


# Loss functions
def gradient_penalty(samples, output, weight):
    gradients = K.gradients(output, samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_pen = K.sum(gradients_sqr,
                         axis=np.arange(1, len(gradients_sqr.shape)))

    # (weight / 2) * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_pen) * weight


class TrainingSession(object):

    def __init__(self,
                 session_id,
                 max_steps=1000000,
                 lr=0.0001,
                 batch_size=16,
                 data_dir='vfp_128',
                 const_seed=1,
                 latent_size=512,
                 img_size=128,
                 n_map_layers=4,
                 fmap_min=8,
                 fmap_max=512,
                 fmap_scale=8):

        self.session_id = session_id

        self.dis = None
        self.map = None
        self.gen = None
        self.gen_eval = None

        self.gen_opt = None
        self.dis_opt = None

        self.config_path = os.path.join('config', self.session_id + '.json')

        # Session config already exists
        if os.path.exists(self.config_path):
            with open(self.config_path) as config_file:
                self.config = json.load(config_file)
        else:
            self.config = {'id': self.session_id,  # Training parameters
                           'steps': 0,
                           'max_steps': max_steps,
                           'lr': lr,
                           'batch_size': batch_size,
                           'data_dir': data_dir,
                           'const_seed': const_seed,
                           'latent_size': latent_size,  # Model parameters
                           'img_size': img_size,
                           'n_map_layers': n_map_layers,
                           'fmap_min': fmap_min,
                           'fmap_max': fmap_max,
                           'fmap_scale': fmap_scale}

        self.steps = self.config['steps']
        self.max_steps = self.config['max_steps']
        self.lr = self.config['lr']
        self.batch_size = self.config['batch_size']
        self.data_dir = self.config['data_dir']
        self.const_seed = self.config['const_seed']

        self.gan = StyleGAN(latent_size=self.config['latent_size'],
                            img_size=self.config['img_size'],
                            n_map_layers=self.config['n_map_layers'],
                            fmap_min=self.config['fmap_min'],
                            fmap_max=self.config['fmap_max'],
                            fmap_scale=self.config['fmap_scale'])

        if self.steps == 0:
            self.dis = self.gan.build_dis()
            self.map = self.gan.build_map()
            self.gen, _ = self.gan.build_gen()
            self.save(0)
        else:
            self.dis = util.load_model('dis', self.steps, self.session_id)
            self.map = util.load_model('map', self.steps, self.session_id)
            self.gen = util.load_model('gen', self.steps, self.session_id)

        self.gen_eval = self.gan.build_gen_eval(self.map, self.gen)

        self.dis_opt = Adam(lr=self.lr, beta_1=0, beta_2=0.999)
        self.gen_opt = Adam(lr=self.lr, beta_1=0, beta_2=0.999)

        self.im = dataGenerator(self.data_dir, self.config['img_size'], flip=True)

        self.last_blip = time.clock()

        self.ones = np.ones((self.batch_size, 1), dtype=np.float32)
        self.zeros = np.zeros((self.batch_size, 1), dtype=np.float32)
        self.nones = -self.ones

        self.pl_mean = 0
        self.av = np.zeros([44])

        self.n_layers = self.gan.n_blocks * 2

    def train(self):

        if random() < mixed_prob:
            style = self.gan.mixed_latent_list(self.batch_size)
        else:
            style = self.gan.random_latent_list(self.batch_size)

        apply_gradient_penalty = self.steps % 2 == 0 or self.steps < 10000
        apply_path_penalty = self.steps % 16 == 0

        a, b, c, d = self.train_step(self.im.get_batch(self.batch_size).astype('float32'),
                                     style,
                                     apply_gradient_penalty,
                                     apply_path_penalty)

        if self.pl_mean == 0:
            self.pl_mean = np.mean(d)
        self.pl_mean = 0.99*self.pl_mean + 0.01*np.mean(d)

        if np.isnan(a):
            print("NaN Value Error.")
            exit()

        if self.steps % 100 == 0:
            print("\n\nRound " + str(self.steps) + ":")
            print("D:", np.array(a))
            print("G:", np.array(b))
            print("PL:", self.pl_mean)

            s = round((time.clock() - self.last_blip), 4)
            self.last_blip = time.clock()

            steps_per_second = 100 / s
            steps_per_minute = steps_per_second * 60
            steps_per_hour = steps_per_minute * 60
            print("Steps/Second: " + str(round(steps_per_second, 2)))
            print("Steps/Hour: " + str(round(steps_per_hour)))

            min1k = floor(1000/steps_per_minute)
            sec1k = floor(1000/steps_per_second) % 60
            print("1k Steps: " + str(min1k) + ":" + str(sec1k))
            steps_left = self.max_steps - self.steps
            hours_left = steps_left // steps_per_hour
            minutes_left = (steps_left // steps_per_minute) % 60

            print("Til Completion: " + str(int(hours_left)) + "h" + str(int(minutes_left)) + "m")
            print()

            if self.steps % 500 == 0:
                self.save(self.steps)
                print('Saved')
            if self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                self.evaluate(self.steps)

        self.steps = self.steps + 1

    @tf.function
    def train_step(self, images, latents, perform_gp=True, perform_pl=False):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            noise = self.gan.random_noise(self.batch_size)

            w_space = []
            pl_lengths = self.pl_mean
            for i in range(len(latents)):
                w_space.append(self.map(latents[i]))

            generated_images = self.gen(w_space + noise + [np.asarray([self.const_seed])])

            # Discriminate
            real_output = self.dis(images, training=True)
            fake_output = self.dis(generated_images, training=True)

            # Hinge loss function
            gen_loss = K.mean(fake_output)
            divergence = K.mean(K.relu(1 + real_output) + K.relu(1 - fake_output))
            disc_loss = divergence

            if perform_gp:
                # R1 gradient penalty
                disc_loss += gradient_penalty(images, real_output, 10)

            if perform_pl:
                w_space_2 = []
                for i in range(len(latents)):
                    std = 0.1 / (K.std(w_space[i], axis=0, keepdims=True) + 1e-8)
                    w_space_2.append(w_space[i] + K.random_normal(tf.shape(w_space[i])) / (std + 1e-8))

                pl_images = self.gen(w_space_2 + noise + [np.asarray([self.const_seed])])

                delta_g = K.mean(K.square(pl_images - generated_images), axis=[1, 2, 3])
                pl_lengths = delta_g

                if self.pl_mean > 0:
                    gen_loss += K.mean(K.square(pl_lengths - self.pl_mean))

        gradients_of_generator = gen_tape.gradient(gen_loss, self.gen_eval.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.dis.trainable_variables)

        self.gen_opt.apply_gradients(zip(gradients_of_generator, self.gen_eval.trainable_variables))
        self.dis_opt.apply_gradients(zip(gradients_of_discriminator, self.dis.trainable_variables))

        return disc_loss, gen_loss, divergence, pl_lengths

    def evaluate(self, num=0, trunc=1.0):

        n1 = self.gan.random_latent_list(64)
        n2 = self.gan.random_noise(64)

        generated_images = self.gen_eval.predict(n1 + n2 + [np.asarray([self.const_seed] * 64)],
                                                 batch_size=self.batch_size)

        r = []

        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i+8], axis=1))

        c1 = np.concatenate(r, axis=0)
        c1 = np.clip(c1, 0.0, 1.0)

        util.save_image(c1, 'i', num, self.session_id)

    def save(self, num):  # Save JSON and Weights into /Models/
        util.save_model(self.map, "map", num, self.session_id)
        util.save_model(self.gen, "gen", num, self.session_id)
        util.save_model(self.dis, "dis", num, self.session_id)

        self.config['steps'] = self.steps
        with open(self.config_path, 'w') as config_file:
            config_file.write(json.dumps(self.config, indent=4))

    def complete(self):
        return self.steps >= self.max_steps


if __name__ == "__main__":

    model = TrainingSession('test4_res512',
                            img_size=512,
                            data_dir='vfp_512',
                            n_map_layers=8,
                            batch_size=16)
    model.map.summary()
    model.gen.summary()
    model.dis.summary()

    while not model.complete():
        model.train()
