

from utils.plot import get_marker_style
from src.model import ConditionalGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplib
from tensorflow.keras import backend as K
import imutils
from tensorflow.keras.preprocessing.image import img_to_array

def display(image, flip=True, cmap=None, figsize=(6, 6), **kwargs):
    if flip:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # adjustment for histograms
    if kwargs.get("is_histogram", False):
        plt.figure(figsize=figsize)
        plt.plot(image)
    else:
        width=image.shape[1]
        height=image.shape[0]
        margin=50
        # dots per inch
        dpi=100.
        # inches
        figsize=((width+2*margin)/dpi, (height+2*margin)/dpi)
        #axes ratio
        left = margin/dpi/figsize[0]
        bottom = margin/dpi/figsize[1]
        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom)
        _ = plt.imshow(image, cmap=cmap)
        plt.axis("off")
    plt.title(kwargs.get("title", None))
    plt.xlabel(kwargs.get("xlabel", None))
    plt.ylabel(kwargs.get("ylabel", None))
    plt.xlim(kwargs.get("xlim", None))
    plt.show()

gen = ConditionalGenerator(
    config="",
    global_step=None
)



image = cv2.cvtColor(cv2.imread("small.png"), cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (150, 150))

images = np.array([image])

print(images.shape)

max_size = list(images.shape)[1:3]
assert max_size[0] == max_size[1]
max_size = max_size[0]

# determine the sizes for the renderer
render_sizes = []
size = max_size
stride = 2
while True:
    render_sizes.append(size)
    if size <= 16:
        break
    size = size // stride

print(render_sizes)

# encodings = gen.pose_encoder(x=images, n_maps=10, map_sizes=render_sizes)
# encodings = gen.image_encoder(x=images)
encodings = gen.encoder(x=images)[-1]

print("ENCODING DONE")

x = gen.conv(
    encodings, filters=10, kernel_size=(1, 1), padding="same",
    strides=1, batch_norm=False, activation=None, name="conv_1"
)

def get_coordinates(x, axis, axis_size):
    # get x,y coordinates
    # B, W, NMAP
    g_c_prob = K.mean(x, axis)
    # B, W, NMAP
    g_c_prob = K.softmax(g_c_prob, axis=1)
    print(axis_size)
    coord_pt = np.linspace(-1.0, 1.0, axis_size)
    coord_pt = np.reshape(coord_pt, (1, axis_size, 1))
    g_c = K.sum(g_c_prob*coord_pt, axis=1)
    return g_c, g_c_prob
"""
y = y.numpy()
y = y.reshape(19, 19, 10)
k = np.split(y, 10, axis=2)
k1 = k[0].reshape(list(k[0].shape)[:2])
print(type(k[1]))
print(k[1].shape)
display(k1, flip=False, cmap="gray")
"""

x_shape = list(x.shape)
print(x_shape)

gauss_y, gauss_y_prob = get_coordinates(x, 2, x_shape[1])
gauss_x, gauss_x_prob = get_coordinates(x, 1, x_shape[2])

print(type(gauss_y))
print(len(gauss_y))
print(gauss_y)
print(type(gauss_y_prob))
print(len(gauss_y_prob))
print(gauss_y_prob)


#gauss_x, gauss_x_prob = get_coordinates(x, 1, x_shape[2])
#gauss_mu = K.stack([gauss_y, gauss_x], axis=2)



# print(type(encodings[0]))
#print(len(encodings[0]))


#target = encodings[0]
#print(target.shape)

# display(target.reshape(list(target.shape)[1:]), flip=False)






















"""

def plot_landmark(ax, landmark, k, size=1.5, zorder=2, cmap='Dark2',
                  style_fn=None):
    
    c, m = get_marker_style(k, cmap=cmap)
    plt.scatter(landmark[1], landmark[0], c=c, marker=m,
               s=(size * mplib.rcParams['lines.markersize']) ** 2,
               zorder=zorder)
    plt.show()
    

def plot_landmarks( landmarks, size=1.5, zorder=2, cmap='Dark2', style_fn=None):
    for k, landmark in enumerate(landmarks):
        plot_landmark(None, landmark, k, size=size, zorder=zorder,
                      cmap=cmap, style_fn=style_fn)


for i, x in enumerate(encodings[1]):
    print(x.shape)
    if i == 4:
        image = x.numpy()
        print(image.shape)
        print(type(image))
        
        # display(image.reshape(image.shape[1:]), flip=False)


landmarks = encodings[1][3].numpy()
#landmarks = encodings[1]
# print(landmarks.shape)

plot_landmarks(
    landmarks=landmarks
)

# [print(encodings[i].shape) for i in range(len(encodings))]
# print(encodings[0].shape)
# display(encodings[0].reshape(encodings[0].shape[1:]), flip=False, figsize=(5, 5))
"""
