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
import glob

# import warnings
# warnings.filterwarnings('ignore')

def main():
    #================================ settings:
    train_the_embedding_space = True
    evaluate_the_embedding_space = False
    assert train_the_embedding_space != evaluate_the_embedding_space
    deep_model = "ResNet"  #--> "CNN", "ResNet"
    loss_type = "easy_positive_triplet_withInnerProduct"   #--> batch_hard_triplet, batch_semi_hard_triplet, batch_all_triplet, Nearest_Nearest_batch_triplet
                                                  #    Nearest_Furthest_batch_triplet, Furthest_Furthest_batch_triplet, Different_distances_batch_triplet
                                                # Negative_sampling_batch_triplet, easy_positive_triplet, Proxy_NCA_triplet_CentersAsProxies, easy_positive_triplet_withInnerProduct
    n_res_blocks = 18  #--> 18, 34, 50, 101, 152
    batch_size = 45  # batch_size must be the same as batch_size in the code of generating batches
    # learning_rate = 0.00001
    learning_rate = 1e-5
    margin_in_loss = 0.25
    feature_space_dimension = 128
    n_triplets_per_batch = 45  #--> n_samples / n_batches --> 15030 / 334
    n_samples_per_class_in_batch = 5  #--> batch_size / n_classes --> 45 / 9
    n_classes = len(dataset_characteristics.get_class_names())
    path_save_network_model = ".\\network_model\\" + deep_model + "\\"
    model_dir_ = model_dir(model_name=deep_model, n_res_blocks=n_res_blocks, batch_size=batch_size, learning_rate=learning_rate)
    #================================ 
    if train_the_embedding_space:
        train_embedding_space(deep_model, n_res_blocks, batch_size, learning_rate, path_save_network_model, model_dir_, feature_space_dimension, margin_in_loss, loss_type,
                              n_triplets_per_batch, n_samples_per_class_in_batch, n_classes)
    if evaluate_the_embedding_space:
        evaluate_embedding_space(path_save_network_model, model_dir_, deep_model, feature_space_dimension, n_res_blocks, margin_in_loss, loss_type,
                                 n_triplets_per_batch, n_samples_per_class_in_batch, n_classes, batch_size)

def evaluate_embedding_space(path_save_network_model, model_dir_, deep_model, feature_space_dimension, n_res_blocks, margin_in_loss, loss_type,
                             n_triplets_per_batch, n_samples_per_class_in_batch, n_classes, batch_size):
    which_epoch_to_load_NN_model = 50
    batch_size_test = 100
    task_to_do = "classify"   #--> read_into_batches, embed_test_data, classify
    proportions = [0.05, 0.1, 0.25, 0.5, 1]

    path_dataset_test = "D:\\Datasets\\CRC_new_large\\CRC_100K_train_test_numpy\\test2"
    path_save_test_patches = ".\\results\\" + deep_model + "\\batches_test2_set\\"
    path_save_embeddings_of_test_data = ".\\results\\" + deep_model + "\\embedding_test2_set\\"
    path_save_accuracy_of_test_data = ".\\results\\" + deep_model + "\\accuracy_test2_set\\"
    image_height = dataset_characteristics.get_image_height()
    image_width = dataset_characteristics.get_image_width()
    image_n_channels = dataset_characteristics.get_image_n_channels()

    if deep_model == "CNN":
        siamese = CNN_Siamese.CNN_Siamese(loss_type=loss_type, feature_space_dimension=feature_space_dimension, margin_in_loss=margin_in_loss)
    elif deep_model == "ResNet":
        # siamese = ResNet_Siamese.ResNet_Siamese(loss_type=loss_type, feature_space_dimension=feature_space_dimension,
        #                                         n_res_blocks=n_res_blocks, margin_in_loss=margin_in_loss, is_train=True)
        siamese = ResNet_Siamese.ResNet_Siamese(loss_type=loss_type, feature_space_dimension=feature_space_dimension, n_triplets_per_batch=n_triplets_per_batch, 
                                                n_classes=n_classes, n_samples_per_class_in_batch=n_samples_per_class_in_batch,
                                                n_res_blocks=n_res_blocks, margin_in_loss=margin_in_loss, is_train=True, batch_size=batch_size)
    evaluate_ = Evaluate_embedding_space(checkpoint_dir=path_save_network_model+str(which_epoch_to_load_NN_model)+"/", model_dir_=model_dir_, deep_model=deep_model,
                                            batch_size=batch_size_test, feature_space_dimension=feature_space_dimension)

    if task_to_do == "read_into_batches":
        paths_of_images = evaluate_.read_batches_paths(path_dataset=path_dataset_test, path_save_test_patches=path_save_test_patches)
    elif task_to_do == "embed_test_data":
        file = open(path_save_test_patches + 'paths_of_images.pickle', 'rb')
        paths_of_images = pickle.load(file)
        file.close()
        batches, batches_subtypes = evaluate_.read_data_into_batches(paths_of_images=paths_of_images)
        embedding, labels = evaluate_.embed_data_in_the_source_domain(batches=batches, batches_subtypes=batches_subtypes, 
                                                                        siamese=siamese, path_save_embeddings_of_test_data=path_save_embeddings_of_test_data)
    elif task_to_do == "classify":
        embedding = np.load(".\\results\\ResNet\\embedding_test2_set\\numpy\\embedding.npy")
        labels = np.load(".\\results\\ResNet\\embedding_test2_set\\numpy\\subtypes.npy")
        evaluate_.classification_in_target_domain_different_data_portions(X=embedding, y=labels, path_save_accuracy_of_test_data=path_save_accuracy_of_test_data, 
                                                                          proportions=proportions, cv=10)

def train_embedding_space(deep_model, n_res_blocks, batch_size, learning_rate, path_save_network_model, model_dir_, feature_space_dimension, margin_in_loss, loss_type,
                          n_triplets_per_batch, n_samples_per_class_in_batch, n_classes):
    #================================ settings:
    Triplet_type = "Different_Distances"  # "Nearest_Nearest", "Nearest_Furthest", "Furthest_Nearest", "Furthest_Furthest", "Different_Distances", "Regular"
    save_plot_embedding_space = True
    save_points_in_embedding_space = True
    load_saved_network_model = False
    which_epoch_to_load_NN_model = 5
    num_epoch = 51
    save_network_model_every_how_many_epochs = 5
    save_embedding_every_how_many_epochs = 5
    n_samples_plot = 2000   #--> if None, plot all
    image_height = dataset_characteristics.get_image_height()
    image_width = dataset_characteristics.get_image_width()
    image_n_channels = dataset_characteristics.get_image_n_channels()
    path_save_embedding_space = ".\\results\\" + deep_model + "\\embedding_train_set\\"
    path_save_loss = ".\\loss_saved\\"
    # path_batches = "D:\\siamese_considering_distance\\codes\\9_create_batches_for_batchBasedMethods\\code\\batches\\"
    # path_base_data_numpy = "D:\\Datasets\\CRC_new_large\\CRC_100K_train_test_numpy\\test1\\"
    path_batches = "C:\\Users\\bghojogh\\Desktop\\code_pathology\\9_create_batches_for_batchBasedMethods\\code\\batches\\"
    path_base_data_numpy = "C:\\Users\\bghojogh\\Desktop\\code_pathology\\data\\"
    #================================ 

    with open(path_batches + 'batches.pickle', 'rb') as handle:
        loaded_batches_names = pickle.load(handle)
    STEPS_PER_EPOCH_TRAIN = len(loaded_batches_names)  #--> must be the number of batches 

    # Siamese:
    if deep_model == "CNN":
        siamese = CNN_Siamese.CNN_Siamese(loss_type=loss_type, feature_space_dimension=feature_space_dimension, margin_in_loss=margin_in_loss)
    elif deep_model == "ResNet":
        siamese = ResNet_Siamese.ResNet_Siamese(loss_type=loss_type, feature_space_dimension=feature_space_dimension, n_triplets_per_batch=n_triplets_per_batch, 
                                                n_classes=n_classes, n_samples_per_class_in_batch=n_samples_per_class_in_batch,
                                                n_res_blocks=n_res_blocks, margin_in_loss=margin_in_loss, is_train=True, batch_size=batch_size)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(siamese.loss)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(siamese.loss)
    # tf.initialize_all_variables().run()

    saver_ = tf.train.Saver(max_to_keep=None)  # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

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
            embeddings_in_epoch = np.zeros((STEPS_PER_EPOCH_TRAIN * batch_size, feature_space_dimension))
            labels_in_epoch = np.zeros((STEPS_PER_EPOCH_TRAIN * batch_size,))
            for i in range(STEPS_PER_EPOCH_TRAIN):
                if i % 10 == 0:
                    print("STEPS_PER_EPOCH_TRAIN " + str(i) + "/" + str(STEPS_PER_EPOCH_TRAIN) + "...")

                loaded_batch, loaded_labels = read_batches_data(loaded_batch_names=loaded_batches_names[i], batch_size=batch_size, path_base_data_numpy=path_base_data_numpy)

                loaded_batch = loaded_batch.reshape((batch_size, image_height, image_width, image_n_channels))

                _, loss_v, embedding1 = sess.run([train_step, siamese.loss, siamese.o1], feed_dict={siamese.x1: loaded_batch,
                                                                                                    siamese.labels1: loaded_labels})

                embeddings_in_epoch[ ((i*batch_size)+(0*batch_size)) : ((i*batch_size)+(1*batch_size)), : ] = embedding1

                labels_in_epoch[ ((i*batch_size)+(0*batch_size)) : ((i*batch_size)+(1*batch_size)) ] = loaded_labels

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

def read_batches_data(loaded_batch_names, batch_size, path_base_data_numpy):
    # batch_size must be the same as batch_size in the code of generating batches
    tissue_type_list = ["00_TUMOR", "01_STROMA", "02_MUCUS", "03_LYMPHO", "04_DEBRIS", "05_SMOOTH_MUSCLE", "06_ADIPOSE", "07_BACKGROUND", "08_NORMAL"]
    image_height = dataset_characteristics.get_image_height()
    image_width = dataset_characteristics.get_image_width()
    image_n_channels = dataset_characteristics.get_image_n_channels()
    image_n_channels = dataset_characteristics.get_image_n_channels()
    paths_data_files = glob.glob(path_base_data_numpy + "**\\*.npy")
    loaded_batch = np.zeros((batch_size, image_height, image_width, image_n_channels))
    loaded_labels = np.zeros((batch_size,))
    for index_in_batch, file_name in enumerate(loaded_batch_names):
        path_file_in_batch = [i for i in paths_data_files if file_name in i]
        assert len(path_file_in_batch) == 1
        path_ = path_file_in_batch[0]
        class_label = path_.split("\\")[-2]
        class_index = tissue_type_list.index(class_label)
        file_in_batch = np.load(path_)
        loaded_batch[index_in_batch, :, :, :] = file_in_batch
        loaded_labels[index_in_batch] = class_index
    return loaded_batch, loaded_labels

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