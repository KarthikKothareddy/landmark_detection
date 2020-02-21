import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras import backend as K


def softmax(x, axis=-1):
    xt = x/1.0
    e_x = np.exp(xt - np.max(xt, axis=axis, keepdims=True))
    d = np.sum(e_x, axis=axis, keepdims=True)
    return e_x/d


def get_gaussian_maps(mu, shape_hw, inv_std, mode='default'):

    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
    y = np.linspace(-1.0, 1.0, shape_hw[0])
    x = np.linspace(-1.0, 1.0, shape_hw[1])

    if mode in ['rot', 'flat', 'default']:
        mu_y, mu_x = K.expand_dims(mu_y, -1), K.expand_dims(mu_x, -1)
        y = np.reshape(y, (1, 1, shape_hw[0], 1))
        x = np.reshape(x, [1, 1, 1, shape_hw[1]])

        g_y = K.square(y-mu_y)
        g_x = K.square(x-mu_x)
        dist = (g_y + g_x) * inv_std**2

        if mode == "rot":
            g_yx = K.exp(-dist)
        else:
            g_yx = K.exp(-K.pow(dist + 1e-5, 0.25))
    else:
        raise ValueError("Unknown Mode: " + str(mode))
    g_yx = K.permute_dimensions(g_yx, pattern=(0, 2, 3, 1))
    return g_yx


def dev_wrap(fn, dev=None):
    if dev:
        with tf.device(dev):
            x = fn()
    else:
        x = fn()
    return x


def get_n_colors(n, pastel_factor=0.9):
    colors = []
    for i in range(n):
        colors.append(generate_new_color(colors,pastel_factor = 0.9))
    return colors

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])


def get_random_color(pastel_factor=0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]


def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def colorize_landmark_maps(maps):
    n_maps = list(maps.shape)[-1]
    # get n_colors
    colors = get_n_colors(n_maps, pastel_factor=0.0)
    heat_maps = [
        K.expand_dims(maps[..., i], axis=3) * np.reshape(colors[i], (1, 1, 1, 3))
        for i in range(n_maps)
    ]
    return K.max(heat_maps, axis=0)




