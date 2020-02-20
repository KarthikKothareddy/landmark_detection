from tensorflow.keras.layers import (
    Input, Conv2D,
    Flatten, Dense,
    Conv2DTranspose, Reshape,
    Lambda, Activation,
    BatchNormalization, LeakyReLU,
    Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model


def encode(**kwargs):
