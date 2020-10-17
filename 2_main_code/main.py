# required tensoefrlow version: 1.14.0
# conda install -c anaconda tensorflow-gpu==1.14.0

import Utils
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import CNN_Siamese
import ResNet_Siamese
import numpy as np
import matplotlib.pyplot as plt
# import cv2
from collections import OrderedDict  #--> for not repeating legends in plot
import umap
import os
from Evaluate_embedding_space import Evaluate_embedding_space
import dataset_characteristics
import pickle

# import warnings
# warnings.filterwarnings('ignore')

def main():
    #================================ settings:
    train_the_embedding_space = False
    evaluate_the_embedding_space = True
    assert train_the_embedding_space != evaluate_the_embedding_space
    deep_model = "ResNet"  #--> "CNN", "ResNet"
    loss_type = "triplet"   #--> "triplet", "FDA", "contrastive", "FDA_contrastive"
    n_res_blocks = 18  #--> 18, 34, 50, 101, 152
    batch_size = 16
    # learning_rate = 0.00001
    learning_rate = 1e-5
    margin_in_loss = 0.25
    feature_space_dimension = 128
    path_save_network_model = ".\\network_model\\" + deep_model + "\\"
    model_dir_ = model_dir(model_name=deep_model, n_res_blocks=n_res_blocks, batch_size=batch_size, learning_rate=learning_rate)
    #================================ 
    if train_the_embedding_space:
        train_embedding_space(deep_model, n_res_blocks, batch_size, learning_rate, path_save_network_model, model_dir_, feature_space_dimension, margin_in_loss, loss_type)
    if evaluate_the_embedding_space:
        evaluate_embedding_space(path_save_network_model, model_dir_, deep_model, feature_space_dimension, n_res_blocks, margin_in_loss, loss_type)

def evaluate_embedding_space(path_save_network_model, model_dir_, deep_model, feature_space_dimension, n_res_blocks, margin_in_loss, loss_type):
    Triplet_type = "Different_Distances"  # "Nearest_Nearest", "Nearest_Furthest", "Furthest_Nearest", "Furthest_Furthest", "Different_Distances", "Regular"
    which_epoch_to_load_NN_model = 50
    batch_size_test = 100
    read_into_batches_again = False
    embed_test_data_again = True
    if embed_test_data_again:
        assert read_into_batches_again is False

    path_dataset_test = "D:\\Datasets\\CRC_new_large\\CRC_100K_train_test_numpy\\test2"
    path_save_test_patches = ".\\results\\" + deep_model + "\\batches_test2_set\\"
    path_save_embeddings_of_test_data = ".\\results\\" + deep_model + "\\embedding_test2_set\\"
    image_height = dataset_characteristics.get_image_height()
    image_width = dataset_characteristics.get_image_width()
    image_n_channels = dataset_characteristics.get_image_n_channels()

    if deep_model == "CNN":
        siamese = CNN_Siamese.CNN_Siamese(loss_type=loss_type, feature_space_dimension=feature_space_dimension, margin_in_loss=margin_in_loss)
    elif deep_model == "ResNet":
        siamese = ResNet_Siamese.ResNet_Siamese(loss_type=loss_type, feature_space_dimension=feature_space_dimension,
                                                n_res_blocks=n_res_blocks, margin_in_loss=margin_in_loss, is_train=True)
    evaluate_ = Evaluate_embedding_space(checkpoint_dir=path_save_network_model+str(which_epoch_to_load_NN_model)+"/", model_dir_=model_dir_, deep_model=deep_model,
                                            batch_size=batch_size_test, feature_space_dimension=feature_space_dimension)

    if read_into_batches_again:
        paths_of_images = evaluate_.read_batches_paths(path_dataset=path_dataset_test, path_save_test_patches=path_save_test_patches)
    if embed_test_data_again:
        file = open(path_save_test_patches + 'paths_of_images.pickle', 'rb')
        paths_of_images = pickle.load(file)
        file.close()
        batches, batches_subtypes = evaluate_.read_data_into_batches(paths_of_images=paths_of_images)
        embedding, labels = evaluate_.embed_data_in_the_source_domain(batches=batches, batches_subtypes=batches_subtypes, 
                                                                        siamese=siamese, path_save_embeddings_of_test_data=path_save_embeddings_of_test_data)


def train_embedding_space(deep_model, n_res_blocks, batch_size, learning_rate, path_save_network_model, model_dir_, feature_space_dimension, margin_in_loss, loss_type):
    #================================ settings:
    Triplet_type = "Different_Distances"  # "Nearest_Nearest", "Nearest_Furthest", "Furthest_Nearest", "Furthest_Furthest", "Different_Distances", "Regular"
    save_plot_embedding_space = True
    save_points_in_embedding_space = True
    load_saved_network_model = False
    which_epoch_to_load_NN_model = 45
    num_epoch = 51
    save_network_model_every_how_many_epochs = 5
    save_embedding_every_how_many_epochs = 5
    # STEPS_PER_EPOCH_TRAIN = 704
    # STEPS_PER_EPOCH_TRAIN = 1875
    STEPS_PER_EPOCH_TRAIN = 938  #--> 15000 / 16
    n_samples_plot = 2000   #--> if None, plot all
    image_height = dataset_characteristics.get_image_height()
    image_width = dataset_characteristics.get_image_width()
    image_n_channels = dataset_characteristics.get_image_n_channels()
    path_tfrecords_train_base = "D:\\siamese_considering_distance\\codes\\6_create_triplets_with_distances\\8_correct_statistical_test\\triplets\\" + Triplet_type + "\\"
    if Triplet_type == "Nearest_Nearest":
        path_tfrecords_train = path_tfrecords_train_base + "triplets_Nearest_Nearest.tfrecords"
    elif Triplet_type == "Nearest_Furthest":
        path_tfrecords_train = path_tfrecords_train_base + "triplets_Nearest_Furthest.tfrecords"
    elif Triplet_type == "Furthest_Nearest":
        path_tfrecords_train = path_tfrecords_train_base + "triplets_Furthest_Nearest.tfrecords"
    elif Triplet_type == "Furthest_Furthest":
        path_tfrecords_train = path_tfrecords_train_base + "triplets_Furthest_Furthest.tfrecords"
    elif Triplet_type == "Different_Distances":
        path_tfrecords_train = path_tfrecords_train_base + "triplets_Different_Distances.tfrecords"
    elif Triplet_type == "Regular":
        path_tfrecords_train = path_tfrecords_train_base + "triplets_Regular.tfrecords"
    path_save_embedding_space = ".\\results\\" + deep_model + "\\embedding_train_set\\"
    path_save_loss = ".\\loss_saved\\"
    #================================ 

    train_dataset = tf.data.TFRecordDataset([path_tfrecords_train])
    train_dataset = train_dataset.map(Utils.parse_function)
    train_dataset = train_dataset.map(Utils.normalize_triplets)

    num_repeat = None
    train_dataset = train_dataset.repeat(num_repeat)
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(batch_size)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types,
                                                             train_dataset.output_shapes)

    next_element = iterator.get_next()
    # training_iterator = train_dataset.make_initializable_iterator()
    training_iterator = tf.data.make_initializable_iterator(train_dataset)

    # Siamese:
    if deep_model == "CNN":
        siamese = CNN_Siamese.CNN_Siamese(loss_type=loss_type, feature_space_dimension=feature_space_dimension, margin_in_loss=margin_in_loss)
    elif deep_model == "ResNet":
        siamese = ResNet_Siamese.ResNet_Siamese(loss_type=loss_type, feature_space_dimension=feature_space_dimension,
                                                n_res_blocks=n_res_blocks, margin_in_loss=margin_in_loss, is_train=True)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(siamese.loss)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(siamese.loss)
    # tf.initialize_all_variables().run()

    saver_ = tf.train.Saver(max_to_keep=None)  # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        training_handle = sess.run(training_iterator.string_handle())
        sess.run(training_iterator.initializer)

        if load_saved_network_model:
            succesful_load, latest_epoch = load_network_model(saver_=saver_, session_=sess, checkpoint_dir=path_save_network_model+str(which_epoch_to_load_NN_model)+"/",
                                                                model_dir_=model_dir_, model_name=deep_model)
            assert (succesful_load == True)
            loss_average_of_epochs = np.load(path_save_loss + "loss.npy")
            loss_average_of_epochs = loss_average_of_epochs[:latest_epoch+1]
            loss_average_of_epochs = list(loss_average_of_epochs)
        else:
            latest_epoch = -1
            loss_average_of_epochs = []

        for epoch in range(latest_epoch+1, num_epoch):
            losses_in_epoch = []
            print("============= epoch: " + str(epoch) + "/" + str(num_epoch-1))
            embeddings_in_epoch = np.zeros((STEPS_PER_EPOCH_TRAIN * batch_size * 3, feature_space_dimension))
            labels_in_epoch = np.zeros((STEPS_PER_EPOCH_TRAIN * batch_size * 3,))
            for i in range(STEPS_PER_EPOCH_TRAIN):
                if i % 10 == 0:
                    print("STEPS_PER_EPOCH_TRAIN " + str(i) + "/" + str(STEPS_PER_EPOCH_TRAIN) + "...")
                image_anchor, image_neighbor, image_distant, label_anchor, label_neighbor, label_distant = sess.run(next_element,
                                                                       feed_dict={handle: training_handle})

                image_anchor = image_anchor.reshape((batch_size, image_height, image_width, image_n_channels))
                image_neighbor = image_neighbor.reshape((batch_size, image_height, image_width, image_n_channels))
                image_distant = image_distant.reshape((batch_size, image_height, image_width, image_n_channels))

                _, loss_v, embedding1, embedding2, embedding3 = sess.run([train_step, siamese.loss, siamese.o1, siamese.o2, siamese.o3], feed_dict={
                                                                                                    siamese.x1: image_anchor,
                                                                                                    siamese.x2: image_neighbor,
                                                                                                    siamese.x3: image_distant})

                embeddings_in_epoch[ ((i*3*batch_size)+(0*batch_size)) : ((i*3*batch_size)+(1*batch_size)), : ] = embedding1
                embeddings_in_epoch[ ((i*3*batch_size)+(1*batch_size)) : ((i*3*batch_size)+(2*batch_size)), : ] = embedding2
                embeddings_in_epoch[ ((i*3*batch_size)+(2*batch_size)) : ((i*3*batch_size)+(3*batch_size)), : ] = embedding3

                labels_in_epoch[ ((i*3*batch_size)+(0*batch_size)) : ((i*3*batch_size)+(1*batch_size)) ] = label_anchor
                labels_in_epoch[ ((i*3*batch_size)+(1*batch_size)) : ((i*3*batch_size)+(2*batch_size)) ] = label_neighbor
                labels_in_epoch[ ((i*3*batch_size)+(2*batch_size)) : ((i*3*batch_size)+(3*batch_size)) ] = label_distant

                losses_in_epoch.extend([loss_v])
                
            # report average loss of epoch:
            loss_average_of_epochs.append(np.average(np.asarray(losses_in_epoch)))
            print("Average loss of epoch " + str(epoch) + ": " + str(loss_average_of_epochs[-1]))
            if not os.path.exists(path_save_loss):
                os.makedirs(path_save_loss)
            np.save(path_save_loss + "loss.npy", np.asarray(loss_average_of_epochs))

            # plot the embedding space:
            if (epoch % save_embedding_every_how_many_epochs == 0):
                if save_points_in_embedding_space:
                    if not os.path.exists(path_save_embedding_space+"numpy\\"):
                        os.makedirs(path_save_embedding_space+"numpy\\")
                    np.save(path_save_embedding_space+"numpy\\embeddings_in_epoch_" + str(epoch) + ".npy", embeddings_in_epoch)
                    np.save(path_save_embedding_space+"numpy\\labels_in_epoch_" + str(epoch) + ".npy", labels_in_epoch)
                if save_plot_embedding_space:
                    print("saving the plot of embedding space....")
                    plt.figure(200)
                    # fig.clf()
                    _, indices_to_plot = plot_embedding_of_points(embeddings_in_epoch, labels_in_epoch, n_samples_plot)
                    if not os.path.exists(path_save_embedding_space+"plots\\"):
                        os.makedirs(path_save_embedding_space+"plots\\")
                    plt.savefig(path_save_embedding_space+"plots\\" + 'epoch' + str(epoch) + '_step' + str(i) + '.png')
                    plt.clf()
                    plt.close()

            # save the network model:
            if (epoch % save_network_model_every_how_many_epochs == 0):
                # save_network_model(saver_=saver_, session_=sess, checkpoint_dir=path_save_network_model, step=epoch, model_name=deep_model, model_dir_=model_dir_)
                save_network_model(saver_=saver_, session_=sess, checkpoint_dir=path_save_network_model+str(epoch)+"/", step=epoch, model_name=deep_model, model_dir_=model_dir_)
                print("Model saved in path: %s" % path_save_network_model)

def plot_embedding_of_points(embedding, labels, n_samples_plot=None):
    n_samples = embedding.shape[0]
    if n_samples_plot != None:
        indices_to_plot = np.random.choice(range(n_samples), min(n_samples_plot, n_samples), replace=False)
    else:
        indices_to_plot = np.random.choice(range(n_samples), n_samples, replace=False)
    embedding_sampled = embedding[indices_to_plot, :]
    if embedding.shape[1] == 2:
        pass
    else:
        embedding_sampled = umap.UMAP(n_neighbors=500).fit_transform(embedding_sampled)
    n_points = embedding.shape[0]
    # n_points_sampled = embedding_sampled.shape[0]
    labels_sampled = labels[indices_to_plot]
    _, ax = plt.subplots(1, figsize=(14, 10))
    classes = dataset_characteristics.get_class_names()
    n_classes = len(classes)
    plt.scatter(embedding_sampled[:, 0], embedding_sampled[:, 1], s=10, c=labels_sampled, cmap='Spectral', alpha=1.0)
    # plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
    cbar.set_ticks(np.arange(n_classes))
    cbar.set_ticklabels(classes)
    return plt, indices_to_plot

def save_network_model(saver_, session_, checkpoint_dir, step, model_name, model_dir_):
    # https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
    # https://github.com/taki0112/ResNet-Tensorflow/blob/master/ResNet.py
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir_)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver_.save(session_, os.path.join(checkpoint_dir, model_name+'.model'), global_step=step)

def load_network_model(saver_, session_, checkpoint_dir, model_dir_, model_name):
    # https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir_)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver_.restore(session_, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        latest_epoch = int(ckpt_name.split("-")[-1])
        return True, latest_epoch
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def model_dir(model_name, n_res_blocks, batch_size, learning_rate):
    return "{}_{}_{}_{}".format(model_name, n_res_blocks, batch_size, learning_rate)


if __name__ == "__main__":
    main()