
import tensorflow as tf


def pairwise_distances(embeddings, squared=False, l2_normalization=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    if l2_normalization:
        embeddings = tf.math.l2_normalize(embeddings)

    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def calculate_positive_and_negative_distance_matrices_in_NCA(labels, embeddings, batch_size, squared=False):
    """
    Returns:
    anchor_positive_dist: a matrix with size (batch_size, batch_size) with only one 1 per row (for the selected positive)
    anchor_negative_dist: a matrix with size (batch_size, batch_size) with entries 1 for negatives
    """
    # Get the pairwise distance matrix
    pairwise_dist = pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)
    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)  #--> this is all anchor-positive distances

    # randomly selecting a positive per anchor:
    n_positives_per_anchor = tf.reduce_sum(mask_anchor_positive, axis=1)
    n_positives_per_anchor = tf.reshape(n_positives_per_anchor, shape=[batch_size,1])
    n_positives_per_anchor_tiled = tf.tile(n_positives_per_anchor, tf.constant([1, batch_size], tf.int32))
    probabilities_of_positives = tf.math.divide(mask_anchor_positive, n_positives_per_anchor_tiled)  #--> zero entries are for either anchors (diagonal) or negatives
    # --> google it: np random choice equivalent tensorflow
    # https://stackoverflow.com/questions/41123879/numpy-random-choice-in-tensorflow
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/multinomial
    # https://www.tensorflow.org/api_docs/python/tf/random/categorical
    selected_indices = tf.multinomial(logits=tf.math.log(probabilities_of_positives), num_samples=1)
    # one hot encoding of the selected_indices to create mask_negative_sampling:
    mask_positive_sampling = tf.one_hot(indices=selected_indices, depth=batch_size)
    mask_positive_sampling = tf.reshape(mask_positive_sampling, shape=[batch_size,batch_size])
    # calculate the anchor_positive_dist by masking:
    anchor_positive_dist = tf.multiply(mask_positive_sampling, anchor_positive_dist) #--> will have one positive per anchor (row)

    # calculating anchor_negative_dist:
    mask_anchor_negative = get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)
    anchor_negative_dist = tf.multiply(mask_anchor_negative, pairwise_dist)

    return anchor_positive_dist, anchor_negative_dist, mask_positive_sampling, mask_anchor_negative


def calculate_proxies(n_classes, feature_space_dimension):
    # https://github.com/dichotomies/proxy-nca/blob/master/proxynca.py
    proxies = tf.random.normal(shape=[n_classes, feature_space_dimension], mean=0.0, stddev=1.0)
    return proxies


def assign_embedding_points_to_proxies(embeddings, proxies, squared=False):
    distance_of_embeddings_from_proxies = distance_of_embeddings(embeddings1=embeddings, embeddings2=proxies, squared=squared, l2_normalization=True)
    proxies_of_points = tf.argmin(distance_of_embeddings_from_proxies, axis=1)
    return proxies_of_points, distance_of_embeddings_from_proxies


def assign_embedding_points_to_proxies_NoNormalization(embeddings, proxies, squared=False):
    distance_of_embeddings_from_proxies = distance_of_embeddings(embeddings1=embeddings, embeddings2=proxies, squared=squared, l2_normalization=False)
    proxies_of_points = tf.argmin(distance_of_embeddings_from_proxies, axis=1)
    return proxies_of_points, distance_of_embeddings_from_proxies


def distance_of_embeddings(embeddings1, embeddings2, squared=False, l2_normalization=False):
    """Compute the 2D matrix of distances between two embeddings.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    if l2_normalization:
        embeddings1 = tf.math.l2_normalize(embeddings1)
        embeddings2 = tf.math.l2_normalize(embeddings2)

    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings1, tf.transpose(embeddings2))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm1 = tf.reduce_sum(tf.math.square(embeddings1), axis=1)
    square_norm2 = tf.reduce_sum(tf.math.square(embeddings2), axis=1)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm1, 1) - 2.0 * dot_product + tf.expand_dims(square_norm2, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def calculate_distances_of_points_from_proxies_of_points(proxies_of_points, distance_of_embeddings_from_proxies, batch_size, n_classes):
    proxies_of_points_oneHot = tf.one_hot(indices=proxies_of_points, depth=n_classes)
    proxies_of_points_oneHot = tf.reshape(proxies_of_points_oneHot, shape=[batch_size,n_classes])
    points_proxOFpoints_dist_TRANSPOSE = tf.linalg.matmul(proxies_of_points_oneHot, tf.transpose(distance_of_embeddings_from_proxies))
    points_proxOFpoints_dist = tf.transpose(points_proxOFpoints_dist_TRANSPOSE)
    return points_proxOFpoints_dist
