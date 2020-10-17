# https://github.com/taki0112/ResNet-Tensorflow

import ops_resnet
import tensorflow as tf
import dataset_characteristics
import random
import numpy as np

class ResNet_Siamese(object):

    def __init__(self, loss_type, feature_space_dimension, n_res_blocks=18, margin_in_loss=0.25, is_train=True, batch_size=32):
        self.img_size_height = dataset_characteristics.get_image_height()
        self.img_size_width = dataset_characteristics.get_image_width()
        self.img_n_channels = dataset_characteristics.get_image_n_channels()
        self.c_dim = 3
        self.res_n = n_res_blocks
        self.feature_space_dimension = feature_space_dimension
        self.margin_in_loss = margin_in_loss
        self.batch_size = batch_size

        self.x1 = tf.placeholder(tf.float32, [None, self.img_size_height, self.img_size_width, self.img_n_channels])
        self.x1Image = self.x1
        self.x2 = tf.placeholder(tf.float32, [None, self.img_size_height, self.img_size_width, self.img_n_channels])
        self.x2Image = self.x2
        self.x3 = tf.placeholder(tf.float32, [None, self.img_size_height, self.img_size_width, self.img_n_channels])
        self.x3Image = self.x3

        self.loss_type = loss_type
        # Create loss
        if is_train:
            if self.loss_type == "triplet":
                with tf.variable_scope("siamese") as scope:
                    self.o1 = self.network(self.x1Image, is_training=True, reuse=False)
                    self.o2 = self.network(self.x2Image, is_training=True, reuse=True)
                    self.o3 = self.network(self.x3Image, is_training=True, reuse=True)
                self.loss = self.loss_triplet()
            elif self.loss_type == "contrastive":
                with tf.variable_scope("siamese") as scope:
                    self.o1 = self.network(self.x1Image, is_training=True, reuse=False)
                    self.o2 = self.network(self.x2Image, is_training=True, reuse=True)
                    self.o3 = self.network(self.x3Image, is_training=True, reuse=True)
                self.loss = self.loss_contrastive()
        else:
            if self.loss_type == "triplet":
                with tf.variable_scope("siamese") as scope:
                    self.o1 = self.network(self.x1Image, is_training=False, reuse=tf.AUTO_REUSE)
            elif self.loss_type == "contrastive":
                with tf.variable_scope("siamese") as scope:
                    self.o1 = self.network(self.x1Image, is_training=False, reuse=tf.AUTO_REUSE)

    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):
            if self.res_n < 50 :
                residual_block = ops_resnet.resblock
            else :
                residual_block = ops_resnet.bottle_resblock

            residual_list = ops_resnet.get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = ops_resnet.conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            ########################################################################################################

            x = ops_resnet.batch_norm(x, is_training, scope='batch_norm')
            x = ops_resnet.relu(x)

            x = ops_resnet.global_avg_pooling(x)
            x = ops_resnet.fully_conneted(x, units=self.feature_space_dimension, scope='logit')

            return x

    def loss_triplet(self):
        d_pos = tf.reduce_sum(tf.square(self.o1 - self.o2), 1)
        d_neg = tf.reduce_sum(tf.square(self.o1 - self.o3), 1)

        loss = tf.maximum(0., self.margin_in_loss + d_pos - d_neg)
        loss = tf.reduce_mean(loss)

        return loss

    def loss_contrastive(self):

        indices_positive = np.random.choice(range(self.batch_size), int(self.batch_size/2), replace=False)
        indices_positive = list(indices_positive)
        indices_negative = [i for i in range(self.batch_size) if i not in indices_positive]
        indices_positive_tf = tf.constant(indices_positive)
        indices_negative_tf = tf.constant(indices_negative)

        # https://stackoverflow.com/questions/38743538/how-to-fetch-specific-rows-from-a-tensor-in-tensorflow
        o1_positive = tf.gather(self.o1, indices=indices_positive_tf, axis=0)
        o2_positive = tf.gather(self.o2, indices=indices_positive_tf, axis=0)

        o1_negative = tf.gather(self.o1, indices=indices_negative_tf, axis=0)
        o3_negative = tf.gather(self.o3, indices=indices_negative_tf, axis=0)

        d_pos = tf.reduce_sum(tf.square(o1_positive - o2_positive), 1)
        d_neg = tf.reduce_sum(tf.square(o1_negative - o3_negative), 1)

        loss = d_pos + tf.maximum(0., self.margin_in_loss - d_neg)

        loss = tf.reduce_mean(loss)

        return loss
