from tensorflow.keras.layers import (
    Input, Conv2D,
    Flatten, Dense,
    Conv2DTranspose, Reshape,
    Lambda, Activation, ReLU,
    BatchNormalization, LeakyReLU,
    Dropout
)
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

import numpy as np
from collections import defaultdict
from utils.utils import get_gaussian_maps, dev_wrap, colorize_landmark_maps


class ConditionalGenerator(object):
    def __init__(self, config, global_step):
        self._config = config
        self._global_step = global_step

    @staticmethod
    def conv(x, filters, kernel_size, strides, name, padding="same", **kwargs):
        # get other args
        batch_norm = kwargs.get("batch_norm", False)
        activation = kwargs.get("activation", ReLU())
        if kwargs.get("weight_decay", False):
            reg = l1(0.95)
        else:
            reg = None
        # conv_2D
        conv_layer = Conv2D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            name=name, kernel_regularizer=reg
        )
        # input = Input(shape=x.shape, name="input")
        # build
        x = x/1.0
        x_ = conv_layer(x)
        # if batch_norm:
        x_ = BatchNormalization()(x_)
        if activation is not None:
            x_ = activation(x_)
        return x_

    def encoder(self, x, **kwargs):
        batch_norm = True
        block_features = []
        # filters = self._config.filters
        filters = 32
        # block 1
        x = self.conv(
            x, filters=filters, kernel_size=(7, 7),
            strides=1, padding="same", name="conv_1", batch_norm=True
        )
        x = self.conv(
            x, filters=filters, kernel_size=(3, 3),
            strides=1, padding="same", name="conv_2", batch_norm=True
        )
        block_features.append(x)
        # block 2
        x = self.conv(
            x, filters=2*filters, kernel_size=(3, 3),
            strides=2, padding="same", name="conv_3", batch_norm=True
        )
        x = self.conv(
            x, filters=2*filters, kernel_size=(3, 3),
            strides=1, padding="same", name="conv_4", batch_norm=True
        )
        block_features.append(x)
        # block 3
        x = self.conv(
            x, filters=4*filters, kernel_size=(3, 3),
            strides=2, padding="same", name="conv_5", batch_norm=True
        )
        x = self.conv(
            x, filters=4*filters, kernel_size=(3, 3),
            strides=1, padding="same", name="conv_6", batch_norm=True
        )
        block_features.append(x)
        # block 3
        x = self.conv(
            x, filters=8*filters, kernel_size=(3, 3),
            strides=2, padding="same", name="conv_7", batch_norm=True
        )
        x = self.conv(
            x, filters=8*filters, kernel_size=(3, 3),
            strides=1, padding="same", name="conv_8", batch_norm=True
        )
        block_features.append(x)
        return block_features

    def image_encoder(self, x, **kwargs):
        # add input to supply max resolution features
        block_features = [x] + self.encoder(x)
        return block_features

    def pose_encoder(self, x, n_maps=1, map_sizes=None, gauss_mode="rot"):
        block_features = self.encoder(x)
        x = block_features[-1]
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
            print(axis_size)
            coord_pt = np.linspace(-1.0, 1.0, axis_size)
            coord_pt = np.reshape(coord_pt, (1, axis_size, 1))
            g_c = K.sum(g_c_prob*coord_pt, axis=1)
            return g_c, g_c_prob

        x_shape = list(x.shape)
        gauss_y, gauss_y_prob = get_coordinates(2, x_shape[1])
        gauss_x, gauss_x_prob = get_coordinates(1, x_shape[2])
        gauss_mu = K.stack([gauss_y, gauss_x], axis=2)
        # return gauss_y, gauss_x

        gauss_xy = []
        for map_size in map_sizes:
            gauss_xy_ = get_gaussian_maps(
                gauss_mu,
                (map_size, map_size),
                # 1.0/self._config.gauss_std,
                1.0/0.10,
                mode=gauss_mode
            )
            gauss_xy.append(gauss_xy_)
        return gauss_mu, gauss_xy


    def simple_renderer(self, feat_hierarchy, n_final_out=3, final_res=128):
        filters = self._config.n_filters_render * 8
        batch_norm = True
        x = feat_hierarchy[16]
        size = list(x.shape)[1:3]
        conv_id = 1
        while size[0] <= final_res:
            x = self.conv(
                x, filters, kernel_size=(3, 3), strides=1,
                batch_norm=batch_norm, name="conv_{}".format(conv_id)
            )
            if size[0] == final_res:
                x = self.conv(
                    x, n_final_out, kernel_size=(3, 3), strides=1,
                    batch_norm=False, name="conv_{}".format(conv_id),
                    activation=None
                )
                break
            size = list(x.shape)[1:3]
            conv_id += 2
            if filters >= 8:
                filters /= 2
        return x

    @staticmethod
    def _loss_mask(map, mask):
        mask = K.resize_images(mask, list(map.shape)[1:3])
        return map*mask

    def model(self, im, future_im, image_encoder, pose_encoder, renderer):
        im_dev, pose_dev, render_dev = None, None, None
        max_size = list(future_im.shape)[1:3]
        assert max_size[0] == max_size[1]
        max_size = max_size[0]
        # determine sizes for rendering
        render_sizes = []
        size = max_size
        stride = self._config.renderer_stride
        while True:
            render_sizes.append(size)
            if size <= self._config.min_res:
                break
            size = size // stride

        embeddings = dev_wrap(lambda: image_encoder(im), im_dev)
        gauss_pt, pose_embeddings = dev_wrap(
            lambda: self.pose_encoder(future_im, map_sizes=render_sizes)
        )

        # helper
        def group_by_size(embeddings):
            # process image embeddings
            grouped_embeddings = defaultdict(list)
            for embedding in embeddings:
                size = list(embedding.shape)[1:3]
                assert size[0] == size[1]
                size = int(size[0])
                grouped_embeddings[size].append(embedding)
            return grouped_embeddings

        grouped_embeddings = group_by_size(embeddings)
        # down sample
        for render_size in render_sizes:
            if render_size not in grouped_embeddings:
                # find closest larger size and resize
                embedding_size = None
                embedding_sizes = sorted(list(grouped_embeddings.keys()))
                for embedding_size in embedding_sizes:
                    if embedding_size >= render_size:
                        break
                resized_embeddings = []
                for embedding in grouped_embeddings[embedding_size]:
                    resized_embeddings.append(
                        K.resize_images(
                            embedding, render_size, render_size,
                            interpolation="bilinear"
                        )
                    )
                grouped_embeddings[render_size] += resized_embeddings

        # process pose embeddings
        grouped_pose_embeddings = group_by_size(pose_embeddings)

        # concatenate embeddings
        joint_embeddings = {}
        for rs in render_sizes:
            joint_embeddings[rs] = K.concatenate(
                grouped_embeddings[rs] + grouped_pose_embeddings[rs],
                axis=1
            )
        future_im_pred = dev_wrap(lambda: renderer(joint_embeddings), render_dev)
        workaround_channels = 0
        color_channels = list(future_im_pred.shape)[3] - workaround_channels
        future_im_pred_mu, _ = np.split(
            future_im_pred, [color_channels, workaround_channels], axis=3
        )
        return future_im_pred_mu, gauss_pt, pose_embeddings

    def build(self, inputs, costs_collection="costs", output_tensors=False, build_loss=True):

        # x, x'
        im, future_im = inputs["image"], inputs["future_image"]

        if "mask" in inputs:
            loss_mask = lambda x: self._loss_mask(x, inputs["mask"])
        else:
            loss_mask = None
        n_maps = self._config.n_maps
        gauss_mode = self._config.gauss_mode
        filters = self._config.n_filters

        future_im_size = list(future_im.shape)[1:3]
        assert future_im_size[0] == future_im_size[1]
        future_im_size = future_im_size[0]
        #
        image_encoder = lambda x: self.image_encoder(x)
        pose_encoder = lambda x, map_sizes, reuse: self.pose_encoder(
            x, n_maps=n_maps, gauss_mode=gauss_mode, map_sizes=map_sizes
        )
        n_renderer_channels = 3
        workaround_channels = 0

        renderer = lambda x: self.simple_renderer(
            x, final_res=future_im_size,
            n_final_out=n_renderer_channels+workaround_channels
        )

        # build the model
        future_im_pred, gauss_yx, pose_embeddings = self.model(
            im,
            future_im,
            image_encoder,
            pose_encoder,
            renderer
        )

        pose_embed_agg = colorize_landmark_maps(pose_embeddings[0])
        future_im_pred_clip = K.clip(future_im_pred,0, 255)

        return pose_embed_agg, future_im_pred_clip




