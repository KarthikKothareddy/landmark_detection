from tensorflow.keras.layers import (
    Input, Conv2D,
    Flatten, Dense,
    Conv2DTranspose, Reshape,
    Lambda, Activation, ReLU,
    BatchNormalization, LeakyReLU,
    Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

import numpy as np
from utils.utils import get_gaussian_maps

class ConditionalGenerator(object):
    def __init__(self, config, global_step):
        self._config = config
        self._global_step = global_step

    def conv(self, x, filters, kernel_size, strides, padding, name, **kwargs):
        # get other args
        batch_norm = kwargs.get("batch_norm", False)
        activation = kwargs.get("activation", ReLU())
        # conv_2D
        conv_layer = Conv2D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            name=name, **kwargs
        )
        # build
        x_ = conv_layer(x)
        x_ = BatchNormalization(x_) if batch_norm else x_
        x_ = activation(x_) if activation is not None else x_
        return x_

    def encoder(self, x, **kwargs):
        batch_norm = True
        block_features = []
        # block 1
        x = self.conv(
            x, filters=self._config.filters, kernel_size=(7, 7),
            strides=1, padding="same", name="conv_1", batch_norm=True
        )
        x = self.conv(
            x, filters=self._config.filters, kernel_size=(3, 3),
            strides=1, padding="same", name="conv_2", batch_norm=True
        )
        block_features.append(x)
        # block 2
        x = self.conv(
            x, filters=2*self._config.filters, kernel_size=(3, 3),
            strides=2, padding="same", name="conv_3", batch_norm=True
        )
        x = self.conv(
            x, filters=2*self._config.filters, kernel_size=(3, 3),
            strides=1, padding="same", name="conv_4", batch_norm=True
        )
        block_features.append(x)
        # block 3
        x = self.conv(
            x, filters=4*self._config.filters, kernel_size=(3, 3),
            strides=2, padding="same", name="conv_5", batch_norm=True
        )
        x = self.conv(
            x, filters=4*self._config.filters, kernel_size=(3, 3),
            strides=1, padding="same", name="conv_6", batch_norm=True
        )
        block_features.append(x)
        # block 3
        x = self.conv(
            x, filters=8*self._config.filters, kernel_size=(3, 3),
            strides=2, padding="same", name="conv_7", batch_norm=True
        )
        x = self.conv(
            x, filters=8*self._config.filters, kernel_size=(3, 3),
            strides=1, padding="same", name="conv_8", batch_norm=True
        )
        block_features.append(x)
        return block_features

    def image_encoder(self, x, **kwargs):
        # add input to supply max resolution features
        block_features = [x] + self.encoder(x)
        return block_features

    def pose_encoder(self, x, n_maps, map_sizes=None, gauss_mode="rot"):
        block_features = self.encoder(x)
        x = block_features[-1]
        x_shape = list(x.shape)
        x = self.conv(
            x, filters=n_maps, kernel_size=(1, 1), padding="same",
            strides=1, batch_norm=False, activation=None, name="conv_1"
        )

        def get_coordinates(axis, axis_size):
            # get x,y coordinates
            # B, W, NMAP
            g_c_prob = K.mean(x, axis)
            # B, W, NMAP
            g_c_prob = K.softmax(g_c_prob, axis=1)
            coord_pt = np.float(np.linspace(-1.0, 1.0, axis_size))
            coord_pt = np.reshape(coord_pt, (1, axis_size, 1))
            g_c = K.sum(g_c_prob*coord_pt, axis=1)
            return g_c, g_c_prob

        x_shape = list(x.shape)
        gauss_y, gauss_y_prob = get_coordinates(2, x_shape[1])
        gauss_x, gauss_x_prob = get_coordinates(1, x_shape[2])
        gauss_mu = K.stack([gauss_y, gauss_x], axis=2)

        gauss_xy = []
        for map_size in map_sizes:
            gauss_xy_ = get_gaussian_maps(
                gauss_mu,
                (map_size, map_size),
                1.0/self._config.gauss_std,
                mode=gauss_mode
            )
            gauss_xy.append(gauss_xy_)
        return gauss_mu, gauss_xy










