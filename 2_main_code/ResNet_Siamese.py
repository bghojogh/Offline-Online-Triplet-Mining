# https://github.com/taki0112/ResNet-Tensorflow

import ops_resnet
import tensorflow as tf
import dataset_characteristics
import random
import numpy as np
import Utils_losses

EPSILON = 1e-10  #--> a very small number

class ResNet_Siamese(object):

    def __init__(self, loss_type, feature_space_dimension, n_triplets_per_batch, n_classes, n_samples_per_class_in_batch, n_res_blocks=18, margin_in_loss=0.25, is_train=True, batch_size=32):
        self.img_size_height = dataset_characteristics.get_image_height()
        self.img_size_width = dataset_characteristics.get_image_width()
        self.img_n_channels = dataset_characteristics.get_image_n_channels()
        self.c_dim = 3
        self.res_n = n_res_blocks
        self.feature_space_dimension = feature_space_dimension
        self.margin_in_loss = margin_in_loss
        self.batch_size = batch_size
        self.n_triplets_per_batch = n_triplets_per_batch
        self.n_classes = n_classes
        self.n_triplets_per_batch_per_class = int(np.floor(self.n_triplets_per_batch / self.n_classes))
        self.n_samples_per_class_in_batch = n_samples_per_class_in_batch

        self.x1 = tf.placeholder(tf.float32, [None, self.img_size_height, self.img_size_width, self.img_n_channels])
        self.x1Image = self.x1
        self.labels1 = tf.placeholder(tf.int32, [None,])

        self.loss_type = loss_type
        # Create loss
        if is_train:
            with tf.variable_scope("siamese") as scope:
                self.o1 = self.network(self.x1Image, is_training=True, reuse=False)
            if self.loss_type == "batch_hard_triplet":
                self.loss = self.batch_hard_triplet_loss(labels=self.labels1, embeddings=self.o1, margin=self.margin_in_loss, squared=True)
            elif self.loss_type == "batch_semi_hard_triplet":
                self.loss = self.batch_semi_hard_triplet_loss(labels=self.labels1, embeddings=self.o1, margin=self.margin_in_loss, squared=True)
            elif self.loss_type == "batch_all_triplet":
                self.loss = self.batch_all_triplet_loss(labels=self.labels1, embeddings=self.o1, margin=self.margin_in_loss, squared=True)
            elif self.loss_type == "Nearest_Nearest_batch_triplet":
                self.loss = self.Nearest_Nearest_batch_triplet_loss(labels=self.labels1, embeddings=self.o1, margin=self.margin_in_loss, squared=True)
            elif self.loss_type == "Nearest_Furthest_batch_triplet":
                self.loss = self.Nearest_Furthest_batch_triplet_loss(labels=self.labels1, embeddings=self.o1, margin=self.margin_in_loss, squared=True)
            elif self.loss_type == "Furthest_Furthest_batch_triplet":
                self.loss = self.Furthest_Furthest_batch_triplet_loss(labels=self.labels1, embeddings=self.o1, margin=self.margin_in_loss, squared=True)
            elif self.loss_type == "Different_distances_batch_triplet":
                self.loss = self.Different_distances_batch_triplet_loss(labels=self.labels1, embeddings=self.o1, margin=self.margin_in_loss, squared=True)
            elif self.loss_type == "Negative_sampling_batch_triplet":
                self.loss = self.Negative_sampling_batch_triplet_loss(labels=self.labels1, embeddings=self.o1, margin=self.margin_in_loss, cutoff=0.5, nonzero_loss_cutoff=1.4, squared=True)
            elif self.loss_type == "NCA_triplet":
                self.loss = self.NCA_triplet_loss(labels=self.labels1, embeddings=self.o1, squared=True)
            elif self.loss_type == "Proxy_NCA_triplet":
                proxies = Utils_losses.calculate_proxies(n_classes=self.n_classes, feature_space_dimension=self.feature_space_dimension)
                self.loss = self.Proxy_NCA_triplet_loss(labels=self.labels1, embeddings=self.o1, proxies=proxies, squared=True)
            elif self.loss_type == "Proxy_NCA_triplet_CentersAsProxies":
                self.loss = self.Proxy_NCA_triplet_loss_CentersAsProxies(labels=self.labels1, embeddings=self.o1, squared=True)
            elif self.loss_type == "easy_positive_triplet":
                self.loss = self.easy_positive_triplet_loss(labels=self.labels1, embeddings=self.o1, squared=True)
            elif self.loss_type == "easy_positive_triplet_withInnerProduct":
                self.loss = self.easy_positive_triplet_loss_withInnerProduct(labels=self.labels1, embeddings=self.o1, squared=True)
        else:
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


    def batch_hard_triplet_loss(self, labels, embeddings, margin, squared=False):  #--> Furthest_Nearest_batch_triplet_loss
        # https://github.com/omoindrot/tensorflow-triplet-loss
        # https://omoindrot.github.io/triplet-loss
        # https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = Utils_losses.pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = Utils_losses.get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

        # shape (batch_size, 1)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = Utils_losses.get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_pairwise_dist_rowwise = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_pairwise_dist_rowwise * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss


    def batch_semi_hard_triplet_loss(self, labels, embeddings, margin, squared=False):
        # https://github.com/omoindrot/tensorflow-triplet-loss
        # https://omoindrot.github.io/triplet-loss
        # https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = Utils_losses.pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = Utils_losses.get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = Utils_losses.get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_pairwise_dist_rowwise = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_pairwise_dist_rowwise * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(anchor_positive_dist - hardest_negative_dist + margin, 0.0)

        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss


    def batch_all_triplet_loss(self, labels, embeddings, margin, squared=False):
        """Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = Utils_losses.pairwise_distances(embeddings, squared=squared)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = Utils_losses.get_triplet_mask(labels)
        mask = tf.to_float(mask)
        triplet_loss = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Get final mean triplet loss:
        triplet_loss = tf.reduce_sum(triplet_loss)

        return triplet_loss


    def Nearest_Nearest_batch_triplet_loss(self, labels, embeddings, margin, squared=False):
        # https://github.com/omoindrot/tensorflow-triplet-loss
        # https://omoindrot.github.io/triplet-loss
        # https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = Utils_losses.pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the extreme positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = Utils_losses.get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

        # We add the maximum value in each row to the invalid positives (valid if a != p and label(a) == label(p))
        max_pairwise_dist_rowwise = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_positive_dist = pairwise_dist + max_pairwise_dist_rowwise * (1.0 - mask_anchor_positive)

        # shape (batch_size, 1)
        extreme_positive_dist = tf.reduce_min(anchor_positive_dist, axis=1, keepdims=True)

        # For each anchor, get the extreme negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = Utils_losses.get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_pairwise_dist_rowwise = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_pairwise_dist_rowwise * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        extreme_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(extreme_positive_dist - extreme_negative_dist + margin, 0.0)

        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss


    def Nearest_Furthest_batch_triplet_loss(self, labels, embeddings, margin, squared=False):
        # https://github.com/omoindrot/tensorflow-triplet-loss
        # https://omoindrot.github.io/triplet-loss
        # https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = Utils_losses.pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the extreme positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = Utils_losses.get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

        # We add the maximum value in each row to the invalid positives (valid if a != p and label(a) == label(p))
        max_pairwise_dist_rowwise = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_positive_dist = pairwise_dist + max_pairwise_dist_rowwise * (1.0 - mask_anchor_positive)

        # shape (batch_size, 1)
        extreme_positive_dist = tf.reduce_min(anchor_positive_dist, axis=1, keepdims=True)

        # For each anchor, get the extreme negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = Utils_losses.get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)

        # We put to 0 any element where (a, p) is not valid (invalid if label(a) == label(n))
        anchor_negative_dist = tf.multiply(mask_anchor_negative, pairwise_dist)

        # shape (batch_size,)
        extreme_negative_dist = tf.reduce_max(anchor_negative_dist, axis=1, keepdims=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(extreme_positive_dist - extreme_negative_dist + margin, 0.0)

        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss


    def Furthest_Furthest_batch_triplet_loss(self, labels, embeddings, margin, squared=False):
        # https://github.com/omoindrot/tensorflow-triplet-loss
        # https://omoindrot.github.io/triplet-loss
        # https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = Utils_losses.pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the extreme positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = Utils_losses.get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

        # shape (batch_size, 1)
        extreme_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

        # For each anchor, get the extreme negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = Utils_losses.get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)

        # We put to 0 any element where (a, p) is not valid (invalid if label(a) == label(n))
        anchor_negative_dist = tf.multiply(mask_anchor_negative, pairwise_dist)

        # shape (batch_size,)
        extreme_negative_dist = tf.reduce_max(anchor_negative_dist, axis=1, keepdims=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(extreme_positive_dist - extreme_negative_dist + margin, 0.0)

        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss


    def Different_distances_batch_triplet_loss(self, labels, embeddings, margin, squared=False):
        # https://github.com/omoindrot/tensorflow-triplet-loss
        # https://omoindrot.github.io/triplet-loss
        # https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = Utils_losses.pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the extreme positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = Utils_losses.get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
        # shape (batch_size, 1)
        extreme_positive_dist_furthest = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

        # We add the maximum value in each row to the invalid positives (valid if a != p and label(a) == label(p))
        max_pairwise_dist_rowwise = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_positive_dist = pairwise_dist + max_pairwise_dist_rowwise * (1.0 - mask_anchor_positive)
        # shape (batch_size, 1)
        extreme_positive_dist_nearest = tf.reduce_min(anchor_positive_dist, axis=1, keepdims=True)

        # For each anchor, get the extreme negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = Utils_losses.get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)

        # We put to 0 any element where (a, p) is not valid (invalid if label(a) == label(n))
        anchor_negative_dist = tf.multiply(mask_anchor_negative, pairwise_dist)
        # shape (batch_size,)
        extreme_negative_dist_furthest = tf.reduce_max(anchor_negative_dist, axis=1, keepdims=True)

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_pairwise_dist_rowwise = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_pairwise_dist_rowwise * (1.0 - mask_anchor_negative)
        # shape (batch_size,)
        extreme_negative_dist_nearest = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

        # randomly take from extreme positive distances:
        ###### create random mask and its logical not:
        mask_random = np.random.randint(2, size=self.batch_size) * 1.0
        mask_random_bool = np.array(mask_random, dtype=bool)
        mask_random_bool_not = np.logical_not(mask_random_bool)
        mask_random_not = np.array(mask_random_bool_not, dtype=int) * 1.0
        mask_random_tensor = tf.convert_to_tensor(mask_random, dtype=tf.float32)
        mask_random_not_tensor = tf.convert_to_tensor(mask_random_not, dtype=tf.float32)
        ###### masking teh distances:
        extreme_positive_dist_furthest_selected = tf.math.multiply(extreme_positive_dist_furthest, mask_random_tensor)
        extreme_positive_dist_nearest_selected = tf.math.multiply(extreme_positive_dist_nearest, mask_random_not_tensor)
        extreme_positive_dist = tf.math.add(extreme_positive_dist_furthest_selected, extreme_positive_dist_nearest_selected)

        # randomly take from extreme negative distances:
        ###### create random mask and its logical not:
        mask_random = np.random.randint(2, size=self.batch_size) * 1.0
        mask_random_bool = np.array(mask_random, dtype=bool)
        mask_random_bool_not = np.logical_not(mask_random_bool)
        mask_random_not = np.array(mask_random_bool_not, dtype=int) * 1.0
        mask_random_tensor = tf.convert_to_tensor(mask_random, dtype=tf.float32)
        mask_random_not_tensor = tf.convert_to_tensor(mask_random_not, dtype=tf.float32)
        ###### masking teh distances:
        extreme_negative_dist_furthest_selected = tf.math.multiply(extreme_negative_dist_furthest, mask_random_tensor)
        extreme_negative_dist_nearest_selected = tf.math.multiply(extreme_negative_dist_nearest, mask_random_not_tensor)
        extreme_negative_dist = tf.math.add(extreme_negative_dist_furthest_selected, extreme_negative_dist_nearest_selected)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(extreme_positive_dist - extreme_negative_dist + margin, 0.0)

        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss


    def Negative_sampling_batch_triplet_loss(self, labels, embeddings, margin, cutoff=0.5, nonzero_loss_cutoff=1.4, squared=False):
        # https://github.com/chaoyuaw/incubator-mxnet/tree/master/example/gluon/embedding_learning
        # https://github.com/chaoyuaw/incubator-mxnet/blob/master/example/gluon/embedding_learning/model.py

        # Get the pairwise distance matrix
        distance = Utils_losses.pairwise_distances(embeddings, squared=squared, l2_normalization=True)

        # Cut off to avoid high variance. Make the distance of the hardest negatives (with small distance from anchor) cliped.
        # distance = tf.clip_by_value(distance, clip_value_min=tf.math.reduce_min(distance), clip_value_max=cutoff)
        distance = tf.clip_by_value(distance, clip_value_min=cutoff, clip_value_max=tf.math.reduce_max(distance))

        # Subtract max(log(distance)) for stability.
        log_weights = ((2.0 - float(self.feature_space_dimension)) * tf.math.log(distance)
                       - (float(self.feature_space_dimension - 3) / 2) * tf.math.log(1.0 - 0.25 * (distance ** 2.0)))
        weights = tf.math.exp(log_weights - tf.math.reduce_max(log_weights))

        # mask the weight matrix to have weights for only negatives:
        mask_anchor_negative = Utils_losses.get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)
        weights_onlyNegatives = tf.multiply(mask_anchor_negative, weights)

        # mask2, ignore the easy negatives (very different from anchor):
        mask2 = tf.math.less(distance, nonzero_loss_cutoff)
        mask2 = tf.to_float(mask2)
        weights = tf.multiply(mask2, weights_onlyNegatives)

        # the weights should be probability:
        # the weights matrix is 2D and non-zero only for negatives and the mask2
        weights_sum_vector = tf.reduce_sum(weights, axis=1)
        weights_sum_vector = tf.reshape(weights_sum_vector, shape=[self.batch_size,1])
        weights_sum_matrix = tf.tile(weights_sum_vector, tf.constant([1, self.batch_size], tf.int32))
        weights = tf.math.divide(weights, weights_sum_matrix) #--> sum of every row becomes one

        # make zero entries of weights epsilon, to prevent NaN in log:
        weights = tf.clip_by_value(weights, clip_value_min=EPSILON, clip_value_max=tf.math.reduce_max(distance))

        # create the mask for negative sampling:
        # --> google it: np random choice equivalent tensorflow
        # https://stackoverflow.com/questions/41123879/numpy-random-choice-in-tensorflow
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/multinomial
        # https://www.tensorflow.org/api_docs/python/tf/random/categorical
        selected_indices = tf.multinomial(logits=tf.math.log(weights), num_samples=1)

        # one hot encoding of the selected_indices to create mask_negative_sampling:
        mask_negative_sampling = tf.one_hot(indices=selected_indices, depth=self.batch_size)
        mask_negative_sampling = tf.reshape(mask_negative_sampling, shape=[self.batch_size,self.batch_size])

        # calculate the anchor_negative_dist by masking:
        anchor_negative_dist = tf.multiply(mask_negative_sampling, distance)

        # calculate the mask_anchor_positive by masking:
        mask_anchor_positive = Utils_losses.get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)
        anchor_positive_dist = tf.multiply(mask_anchor_positive, distance)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = tf.expand_dims(anchor_positive_dist, 2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = tf.expand_dims(anchor_negative_dist, 1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = Utils_losses.get_triplet_mask(labels)
        mask = tf.to_float(mask)
        triplet_loss = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Get final mean triplet loss:
        triplet_loss = tf.reduce_sum(triplet_loss)

        return triplet_loss


    def NCA_triplet_loss(self, labels, embeddings, squared=False):
        # https://github.com/dichotomies/proxy-nca
        # https://github.com/dichotomies/proxy-nca/blob/master/proxynca.py

        anchor_positive_dist, anchor_negative_dist, mask_positive_sampling, mask_anchor_negative = Utils_losses.calculate_positive_and_negative_distance_matrices_in_NCA(labels, embeddings, self.batch_size, squared=squared)

        # softmax:
        numerator_ = tf.math.exp(-1 * anchor_positive_dist)
        denominator_ = tf.reduce_sum(tf.math.exp(-1 * anchor_negative_dist), axis=1, keep_dims=True)
        denominator_ = tf.tile(denominator_, tf.constant([1, self.batch_size], tf.int32))

        denominator_ += EPSILON
        triplet_loss = -1 * tf.math.log(tf.math.divide(numerator_, denominator_) + EPSILON)

        # Get final mean triplet loss:
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss


    def Proxy_NCA_triplet_loss(self, labels, embeddings, proxies, squared=False):
        # https://github.com/dichotomies/proxy-nca
        # https://github.com/dichotomies/proxy-nca/blob/master/proxynca.py

        _, _, mask_positive_sampling, mask_anchor_negative = Utils_losses.calculate_positive_and_negative_distance_matrices_in_NCA(labels, embeddings, self.batch_size, squared=squared)

        proxies_of_points, distance_of_embeddings_from_proxies = Utils_losses.assign_embedding_points_to_proxies(embeddings=embeddings, proxies=proxies, squared=squared)
        points_proxOFpoints_dist = Utils_losses.calculate_distances_of_points_from_proxies_of_points(proxies_of_points, distance_of_embeddings_from_proxies, self.batch_size, self.n_classes)
        anchor_proxOFpositive_dist = tf.multiply(mask_positive_sampling, points_proxOFpoints_dist)
        anchor_proxOFnegative_dist = tf.multiply(mask_anchor_negative, points_proxOFpoints_dist)

        # softmax:
        numerator_ = tf.math.exp(-1 * anchor_proxOFpositive_dist)
        denominator_ = tf.reduce_sum(tf.math.exp(-1 * anchor_proxOFnegative_dist), axis=1, keep_dims=True)
        denominator_ = tf.tile(denominator_, tf.constant([1, self.batch_size], tf.int32))

        denominator_ += EPSILON
        triplet_loss = -1 * tf.math.log(tf.math.divide(numerator_, denominator_) + EPSILON)

        # Get final mean triplet loss:
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss

    def easy_positive_triplet_loss(self, labels, embeddings, squared=False):
        # Paper: Improved Embeddings with Easy Positive Triplet Mining

        # Get the pairwise distance matrix
        pairwise_dist = Utils_losses.pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the extreme positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = Utils_losses.get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

        # We add the maximum value in each row to the invalid positives (valid if a != p and label(a) == label(p))
        max_pairwise_dist_rowwise = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_positive_dist = pairwise_dist + max_pairwise_dist_rowwise * (1.0 - mask_anchor_positive)

        # shape (batch_size, 1)
        extreme_positive_dist = tf.reduce_min(anchor_positive_dist, axis=1, keepdims=True)

        # calculating anchor_negative_dist:
        mask_anchor_negative = Utils_losses.get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)
        anchor_negative_dist = tf.multiply(mask_anchor_negative, pairwise_dist)

        # softmax:
        numerator_ = tf.math.exp(-1 * extreme_positive_dist)
        denominator_ = tf.reduce_sum(tf.math.exp(-1 * anchor_negative_dist), axis=1, keep_dims=True)

        denominator_ += numerator_
        # denominator_ += EPSILON
        triplet_loss = -1 * tf.math.log(tf.math.divide(numerator_, denominator_) + EPSILON)

        # Get final mean triplet loss:
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss


    def easy_positive_triplet_loss_withInnerProduct(self, labels, embeddings, squared=False):
        # Paper: Improved Embeddings with Easy Positive Triplet Mining

        # Get the pairwise distance matrix
        pairwise_dist = Utils_losses.pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the extreme positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = Utils_losses.get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

        # We add the maximum value in each row to the invalid positives (valid if a != p and label(a) == label(p))
        max_pairwise_dist_rowwise = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_positive_dist = pairwise_dist + max_pairwise_dist_rowwise * (1.0 - mask_anchor_positive)

        # calculate anchor_positive_dotProduct:
        easiest_positives_indices = tf.math.argmin(anchor_positive_dist, axis=1)
        easiest_positives_indices_oneHot = tf.one_hot(indices=easiest_positives_indices, depth=self.batch_size)
        easiest_positives_indices_oneHot = tf.reshape(easiest_positives_indices_oneHot, shape=[self.batch_size, self.batch_size])
        embeddings = tf.math.l2_normalize(embeddings)
        dot_product_all_embeddings = tf.matmul(embeddings, tf.transpose(embeddings))
        anchor_positive_dotProduct = tf.reduce_sum(tf.multiply(dot_product_all_embeddings, easiest_positives_indices_oneHot), axis=1) #--> shape (batch_size, 1)

        # calculating anchor_negative_dotProduct:
        mask_anchor_negative = Utils_losses.get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)
        anchor_negative_dotProduct = tf.multiply(dot_product_all_embeddings, mask_anchor_negative) #--> shape (batch_size, batch_size)

        # softmax:
        numerator_ = tf.math.exp(anchor_positive_dotProduct)
        denominator_ = tf.reduce_sum(tf.math.exp(anchor_negative_dotProduct), axis=1, keep_dims=True) #--> shape (batch_size, 1)

        denominator_ += numerator_
        # denominator_ += EPSILON
        triplet_loss = -1 * tf.math.log(tf.math.divide(numerator_, denominator_) + EPSILON)

        # Get final mean triplet loss:
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss


    def Proxy_NCA_triplet_loss_CentersAsProxies(self, labels, embeddings, squared=False):
        # https://github.com/dichotomies/proxy-nca
        # https://github.com/dichotomies/proxy-nca/blob/master/proxynca.py

        # calculate proxies as the center of classes in the batch:
        proxies = tf.Variable(tf.zeros(shape=(self.n_classes, self.feature_space_dimension), dtype=tf.float32))  #--> https://stackoverflow.com/questions/47775067/how-to-assign-the-element-of-tensor-from-other-tensor-in-tensorflow
        for class_index in range(self.n_classes):
            label_of_class = tf.ones(shape=[self.batch_size], dtype=tf.int32) * class_index
            mask_for_samples_of_this_class = tf.math.equal(labels, label_of_class)
            samples_of_this_class = tf.boolean_mask(tensor=embeddings, mask=mask_for_samples_of_this_class, axis=0)
            mean_of_embedding_of_this_class = tf.reduce_mean(samples_of_this_class, axis=0)
            tf.assign(proxies[class_index,:], mean_of_embedding_of_this_class)

        _, _, mask_positive_sampling, mask_anchor_negative = Utils_losses.calculate_positive_and_negative_distance_matrices_in_NCA(labels, embeddings, self.batch_size, squared=squared)

        proxies_of_points, distance_of_embeddings_from_proxies = Utils_losses.assign_embedding_points_to_proxies_NoNormalization(embeddings=embeddings, proxies=proxies, squared=squared)
        points_proxOFpoints_dist = Utils_losses.calculate_distances_of_points_from_proxies_of_points(proxies_of_points, distance_of_embeddings_from_proxies, self.batch_size, self.n_classes)
        anchor_proxOFpositive_dist = tf.multiply(mask_positive_sampling, points_proxOFpoints_dist)
        anchor_proxOFnegative_dist = tf.multiply(mask_anchor_negative, points_proxOFpoints_dist)

        # softmax:
        numerator_ = tf.math.exp(-1 * anchor_proxOFpositive_dist)
        denominator_ = tf.reduce_sum(tf.math.exp(-1 * anchor_proxOFnegative_dist), axis=1, keep_dims=True)
        denominator_ = tf.tile(denominator_, tf.constant([1, self.batch_size], tf.int32))

        denominator_ += EPSILON
        triplet_loss = -1 * tf.math.log(tf.math.divide(numerator_, denominator_) + EPSILON)

        # Get final mean triplet loss:
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss