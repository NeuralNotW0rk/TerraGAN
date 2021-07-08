from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

import numpy as np

data_path = 'data/vfp_256_labeled.npz'


def build_model(features, n_conv_layers, n_fmap, input_shape,
                padding='same',
                activation='relu',
                dropout=0.1):

    model = Sequential()
    model.add(Conv2D(n_fmap[0], kernel_size=3, padding=padding, activation=activation, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    for i in range(1, n_conv_layers):
        model.add(Conv2D(n_fmap[i], kernel_size=3, padding=padding, activation=activation))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(n_fmap[-1]))
    model.add(Dropout(dropout))

    model.add(Dense(features, activation='linear'))

    model.compile(loss='mse', optimizer='adam')

    return model


if __name__ == '__main__':

    model = build_model(2, 6, [32, 64, 128, 256, 256, 256, 256], input_shape=[256, 256, 1])
    model.summary()

    data = np.load(data_path)

    model.fit(x=data['x'], y=data['y'], batch_size=16, epochs=50, validation_split=0.2, verbose=2)

    x = data['x'][0:1]
    y = model.predict(x)

    print(y)
    print(data['y'][0:1])

    model.save('models/min_max_regressor_0.h5')
