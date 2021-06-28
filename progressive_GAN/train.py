from tensorflow.keras.optimizers import Adam

import json

from model import *

LAMBDA = 10


def random_latents(latent_size, n):
    return np.random.normal(0.0, 1.0, size=[n, latent_size])


def load_real_samples(filename):
    data = np.load(filename)
    X = data['arr_0']
    X = X.astype('float32')
    X = (X - 0.5) * 2.0

    return X


class TrainingSession(object):

    def __init__(self,
                 session_id,
                 n_blocks=6,
                 latent_size=100,
                 channels=3,
                 init_res=4,
                 n_fmap=None,
                 block_batch_sizes=None,
                 block_steps=None,
                 data_path='data/vfp_256.npz'):

        self.session_id = session_id

        self.config_path = root_dir + 'config/' + self.session_id + '.json'

        # Session config already exists
        if os.path.exists(self.config_path):
            with open(self.config_path) as config_file:
                self.config = json.load(config_file)
        else:
            sample_latents = []
            for latent in list(random_latents(latent_size, 64)):
                sample_latents.append(list(latent))
            self.config = {'id': self.session_id,
                           'block': 0,
                           'steps': 0,
                           'latent_size': latent_size,
                           'n_fmap': n_fmap,
                           'n_blocks': n_blocks,
                           'block_batch_sizes': list(block_batch_sizes),
                           'block_steps': list(block_steps),
                           'data_path': data_path,
                           'sample_latents': sample_latents}

        self.block = self.config['block']
        self.steps = self.config['steps']
        self.block_batch_sizes = self.config['block_batch_sizes']
        self.block_steps = self.config['block_steps']
        self.data_path = self.config['data_path']
        self.sample_latents = np.asarray(self.config['sample_latents'])

        self.gan = ProgressiveGAN(latent_size=100,
                                  channels=3,
                                  n_blocks=self.block)

        if self.block == 0 and self.steps == 0:
            self.gen = self.gan.build_gen()
            self.dis = self.gan.build_dis()
            self.save()
        else:
            # TODO: Load models
            pass

        self.dis_models = self.gan.build_dis()

        self.gen_opt = Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8)
        self.dis_opt = Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8)

        self.dataset = load_real_samples(data_path)
        self.scaled_dataset = None
        self.scale_dataset()


def update_fadein(models, step, n_steps):
    alpha = step / float(n_steps - 1)
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)


class ArrayTrainingSession(object):

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

        self.config_path = root_dir + 'config/' + self.session_id + '.json'

        # Session config already exists
        if os.path.exists(self.config_path):
            with open(self.config_path) as config_file:
                self.config = json.load(config_file)
        else:
            sample_latents = []
            for latent in list(random_latents(latent_size, 64)):
                sample_latents.append(list(latent))
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
                           'data_path': data_path,
                           'sample_latents': sample_latents}

        self.block = self.config['block']
        self.steps = self.config['steps']
        self.model_version = self.config['model_version']
        self.block_batch_sizes = self.config['block_batch_sizes']
        self.block_steps = self.config['block_steps']
        self.data_path = self.config['data_path']
        self.sample_latents = np.asarray(self.config['sample_latents'])

        self.gan = ProgressiveGANArray(latent_size=self.config['latent_size'],
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

        self.dis_models = self.gan.build_dis()

        self.gen_opt = Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8)
        self.dis_opt = Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8)

        self.dataset = load_real_samples(data_path)
        self.scaled_dataset = None
        self.scale_dataset()

    def real_images(self, n_samples):

        ix = np.random.randint(0, self.scaled_dataset.shape[0], n_samples)
        X = self.scaled_dataset[ix]

        return X

    def scale_dataset(self):

        res = 2 ** (2 + self.block)
        gray = tf.image.resize(self.dataset, [res, res])
        self.scaled_dataset = tf.image.grayscale_to_rgb(gray).numpy()

    # @tf.function
    def train_step(self, images, latents, batch_size):

        generator = self.gen_models[self.block][self.model_version]
        discriminator = self.dis_models[self.block][self.model_version]

        epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)

        with tf.GradientTape(persistent=True) as d_tape:
            with tf.GradientTape() as gp_tape:
                fake_images = generator(latents, training=True)
                fake_images_mixed = epsilon * images + ((1 - epsilon) * fake_images)
                fake_mixed_pred = discriminator(fake_images_mixed, training=True)

            # Compute gradient penalty
            grads = gp_tape.gradient(fake_mixed_pred, fake_images_mixed)
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))

            fake_pred = discriminator(fake_images, training=True)
            real_pred = discriminator(images, training=True)

            D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty

        # Calculate the gradients for discriminator
        D_gradients = d_tape.gradient(D_loss, discriminator.trainable_variables)

        # Apply the gradients to the optimizer
        self.dis_opt.apply_gradients(zip(D_gradients, discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            fake_images = generator(latents, training=True)
            fake_pred = discriminator(fake_images, training=True)
            G_loss = -tf.reduce_mean(fake_pred)

        # Calculate the gradients for discriminator
        G_gradients = g_tape.gradient(G_loss, generator.trainable_variables)

        # Apply the gradients to the optimizer
        self.gen_opt.apply_gradients(zip(G_gradients, generator.trainable_variables))

        return D_loss, G_loss

    def train(self):

        if self.steps > self.block_steps[self.block]:
            if self.block == 0 or self.model_version == 0:
                self.block += 1
                self.model_version = 1
                self.scale_dataset()
                self.steps = 0
            elif self.block == self.gan.n_blocks - 1:
                pass
            else:
                self.model_version = 0
                self.steps = 0

        batch_size = self.block_batch_sizes[self.block]

        gen = self.gen_models[self.block][self.model_version]
        dis = self.dis_models[self.block][self.model_version]

        # Update alpha if needed
        if self.model_version == 1:
            update_fadein([gen, dis], self.steps, self.block_steps[self.block])

        images = self.real_images(batch_size)
        latents = random_latents(self.gan.latent_size, batch_size)

        d_loss, g_loss = self.train_step(images, latents, batch_size)

        if self.steps % 100 == 0:
            print('\nBlock ' + str(self.block)
                  + ', version ' + str(self.model_version)
                  + ', step ' + str(self.steps) + ':')
            print('D:', d_loss.numpy())
            print('G:', g_loss.numpy())

        if self.steps % 10000 == 0:
            self.save()

        if self.steps % 1000 == 0:
            self.evaluate(gen)

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
        imgs = model.predict(self.sample_latents)
        imgs = (imgs + 1.0) / 2.0
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

    ts = ArrayTrainingSession(session_id='pro_test',
                              latent_size=100,
                              n_fmap=128,
                              n_blocks=6,
                              block_batch_sizes=[16, 16, 16, 16, 16, 16],
                              block_steps=[10000, 10000, 10000, 10000, 10000, 10000],
                              data_path='data/vfp_256.npz')

    while not ts.is_complete():
        ts.train()
