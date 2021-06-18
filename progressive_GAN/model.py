'''
Based on code found at https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/
'''

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import RandomNormal

import numpy as np
import matplotlib.pyplot as plt

from layer import *


def layer_name(block_idx, layer_type, layer_idx):
    return 'block' + str(block_idx) + '_' + layer_type + str(layer_idx)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def add_discriminator_block(block_idx, old_model, n_input_layers=3):
    # Initialize params
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)
    in_shape = list(old_model.input.shape)
    input_shape = (in_shape[-2].value * 2, in_shape[-2].value * 2, in_shape[-1].value)

    # New input layer
    in_image = Input(shape=input_shape, name=layer_name(block_idx, 'input', 0))
    d = Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const,
               name=layer_name(block_idx, 'conv', 0))(in_image)
    d = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 0))(d)

    # New block
    d = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
               name=layer_name(block_idx, 'conv', 1))(d)
    d = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 1))(d)
    d = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
               name=layer_name(block_idx, 'conv', 2))(d)
    d = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 2))(d)
    d = AveragePooling2D()(d)
    block_new = d

    # Append new block to old model (skipping old input)
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    model1 = Model(in_image, d)
    model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    # Downsample new input and connect to old model input
    downsample = AveragePooling2D()(in_image)
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)

    # Combine using weighted sum into single straight-through model for fading
    d = WeightedSum()([block_old, block_new])
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    model2 = Model(in_image, d)
    model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    # Return two models: one for the new resolution and one for fading in from the previous resolution
    return [model1, model2]


def define_discriminator(n_blocks, input_shape=(4, 4, 3)):
    # Initialize params
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)

    model_list = list()

    # Base model
    in_image = Input(shape=input_shape, name=layer_name(0, 'input', 0))
    d = Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const,
               name=layer_name(0, 'conv', 0))(in_image)
    d = LeakyReLU(alpha=0.2, name=layer_name(0, 'lrelu', 0))(d)
    d = MinibatchStdev()(d)
    d = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
               name=layer_name(0, 'conv', 1))(d)
    d = LeakyReLU(alpha=0.2, name=layer_name(0, 'lrelu', 1))(d)
    d = Conv2D(128, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const,
               name=layer_name(0, 'conv', 2))(d)
    d = LeakyReLU(alpha=0.2, name=layer_name(0, 'lrelu', 2))(d)
    d = Flatten()(d)
    out_class = Dense(1, name=layer_name(0, 'dense', 0))(d)
    model = Model(in_image, out_class)
    model.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    model_list.append([model, model])

    # Build sub-models
    for i in range(1, n_blocks):
        old_model = model_list[i - 1][0]
        models = add_discriminator_block(i, old_model)
        model_list.append(models)

    return model_list


def add_generator_block(block_idx, old_model):
    # Initialize params
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)

    # Add new block to end of old model
    block_end = old_model.layers[-2].output
    upsampling = UpSampling2D()(block_end)
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
               name=layer_name(block_idx, 'conv', 0))(upsampling)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 0))(g)
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
               name=layer_name(block_idx, 'conv', 1))(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2, name=layer_name(block_idx, 'lrelu', 1))(g)
    out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const,
                       name=layer_name(block_idx, 'to_rgb', 0))(g)
    model1 = Model(old_model.input, out_image)

    # Combine upsampled old output with new output via weighted sum
    out_old = old_model.layers[-1]
    out_image2 = out_old(upsampling)
    merged = WeightedSum()([out_image2, out_image])
    model2 = Model(old_model.input, merged)

    return [model1, model2]


def define_generator(latent_dim, n_blocks, in_dim=4):
    # Initialize parameters
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)

    model_list = list()

    # Base model
    in_latent = Input(shape=(latent_dim,), name=layer_name(0, 'input', 0))
    g = Dense(128 * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const,
              name=layer_name(0, 'dense', 0))(in_latent)
    g = Reshape((in_dim, in_dim, 128))(g)
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
               name=layer_name(0, 'conv', 0))(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2, name=layer_name(0, 'lrelu', 0))(g)
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
               name=layer_name(0, 'conv', 1))(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2, name=layer_name(0, 'irelu', 1))(g)
    out_image = Conv2D(3, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const,
                       name=layer_name(0, 'to_rgb', 0))(g)
    model = Model(in_latent, out_image)
    model_list.append([model, model])

    # Build sub-models
    for i in range(1, n_blocks):
        old_model = model_list[i - 1][0]
        models = add_generator_block(old_model)
        model_list.append(models)

    return model_list


def define_composite(discriminators, generators):
    model_list = list()

    # Build composite models
    for i in range(len(discriminators)):
        g_models, d_models = generators[i], discriminators[i]

        # Straight-through model
        d_models[0].trainable = False
        model1 = Sequential()
        model1.add(g_models[0])
        model1.add(d_models[0])
        model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        # Stage transition model
        d_models[1].trainable = False
        model2 = Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        model_list.append([model1, model2])

    return model_list


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


def train_epochs(g_model, d_model, gan_model, dataset, n_epochs, n_batch, fadein=False):
    # Calculate training hyperparameters
    bat_per_epo = int(dataset.shape[0] / n_batch)
    n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)

    for i in range(n_steps):
        # Update alpha
        if fadein:
            update_fadein([g_model, d_model, gan_model], i, n_steps)

        # Prepare samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

        # Train discriminator
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)

        # Train generator
        z_input = generate_latent_points(latent_dim, n_batch)
        y_real2 = np.ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(z_input, y_real2)

        print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss1, d_loss2, g_loss))


def scale_dataset(images, new_shape):
    images_list = list()
    for image in images:
        new_image = np.resize(image, new_shape, 0)
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
    filename1 = 'results/plot_%s.png' % name
    plt.savefig(filename1)
    plt.close()

    # Save generator model
    filename2 = 'models/model_%s.h5' % name
    g_model.save(filename2)

    print('>Saved: %s and %s' % (filename1, filename2))


def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch):
    # Fit base model
    g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]

    # Scale dataset for base level
    gen_shape = g_normal.output_shape
    scaled_data = scale_dataset(dataset, gen_shape[1:])
    print('Scaled Data', scaled_data.shape)

    # Train base model
    train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], n_batch[0])
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