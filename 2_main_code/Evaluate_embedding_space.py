
import tensorflow as tf
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import dataset_characteristics
import glob

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
import itertools
import dataset_characteristics


class Evaluate_embedding_space():

    def __init__(self, checkpoint_dir, model_dir_, deep_model, batch_size, feature_space_dimension):
        self.checkpoint_dir = checkpoint_dir
        self.model_dir_ = model_dir_
        self.batch_size = batch_size
        self.feature_space_dimension = feature_space_dimension
        self.batch_size = batch_size
        self.n_samples = None
        self.n_batches = None
        self.image_height = dataset_characteristics.get_image_height()
        self.image_width = dataset_characteristics.get_image_width()
        self.image_n_channels = dataset_characteristics.get_image_n_channels()

    def embed_data_in_the_source_domain(self, batches, batches_subtypes, siamese, path_save_embeddings_of_test_data):
        print("Embedding the test set into the source domain....")
        n_batches = int(np.ceil(self.n_samples / self.batch_size))
        saver_ = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            embedding = np.zeros((self.n_samples, self.feature_space_dimension))
            subtypes = [None] * self.n_samples
            for batch_index in range(n_batches):
                print("processing batch " + str(batch_index) + "/" + str(n_batches-1))
                X_batch = batches[batch_index]
                succesful_load, latest_epoch = self.load_network_model(saver_=saver_, session_=sess, checkpoint_dir=self.checkpoint_dir,
                                                                    model_dir_=self.model_dir_)
                assert (succesful_load == True)
                X_batch = self.normalize_images(X_batch)
                test_feed_dict = {
                    siamese.x1: X_batch
                }
                embedding_batch = sess.run(siamese.o1, feed_dict=test_feed_dict)
                if batch_index != (n_batches-1):
                    embedding[(batch_index * self.batch_size) : ((batch_index+1) * self.batch_size), :] = embedding_batch
                    subtypes[(batch_index * self.batch_size) : ((batch_index+1) * self.batch_size)] = batches_subtypes[batch_index]
                else:
                    embedding[(batch_index * self.batch_size) : , :] = embedding_batch
                    subtypes[(batch_index * self.batch_size) : ] = batches_subtypes[batch_index]
            if not os.path.exists(path_save_embeddings_of_test_data+"numpy\\"):
                os.makedirs(path_save_embeddings_of_test_data+"numpy\\")
            np.save(path_save_embeddings_of_test_data+"numpy\\embedding.npy", embedding)
            np.save(path_save_embeddings_of_test_data+"numpy\\subtypes.npy", subtypes)
            if not os.path.exists(path_save_embeddings_of_test_data+"plots\\"):
                os.makedirs(path_save_embeddings_of_test_data+"plots\\")
            # plt.figure(200)
            plt = self.Kather_get_color_and_shape_of_points(embedding=embedding, subtype_=subtypes)
            plt.savefig(path_save_embeddings_of_test_data+"plots\\" + 'embedding.png')
            plt.clf()
            plt.close()
        return embedding, subtypes

    def Kather_get_color_and_shape_of_points(self, embedding, subtype_):
        if embedding.shape[1] == 2:
            embedding_ = embedding
        else:
            embedding_ = umap.UMAP(n_neighbors=500).fit_transform(embedding)
        n_points = embedding_.shape[0]
        labels = np.zeros((n_points,))
        labels[np.asarray(subtype_)=="01_TUMOR"] = 0
        labels[np.asarray(subtype_)=="02_STROMA"] = 1
        labels[np.asarray(subtype_)=="03_COMPLEX"] = 2
        labels[np.asarray(subtype_)=="04_LYMPHO"] = 3
        labels[np.asarray(subtype_)=="05_DEBRIS"] = 4
        labels[np.asarray(subtype_)=="06_MUCOSA"] = 5
        labels[np.asarray(subtype_)=="07_ADIPOSE"] = 6
        labels[np.asarray(subtype_)=="08_EMPTY"] = 7
        _, ax = plt.subplots(1, figsize=(14, 10))
        classes = ["TUMOR", "STROMA", "COMPLEX", "LYMPHO", "DEBRIS", "MUCOSA", "ADIPOSE", "EMPTY"]
        n_classes = len(classes)
        plt.scatter(embedding_[:, 0], embedding_[:, 1], s=10, c=labels, cmap='Spectral', alpha=1.0)
        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
        cbar.set_ticks(np.arange(n_classes))
        cbar.set_ticklabels(classes)
        return plt

    def read_data_into_batches(self, paths_of_images):
        self.n_samples = len(paths_of_images)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        batches = [None] * self.n_batches
        batches_subtypes = [None] * self.n_batches
        for batch_index in range(self.n_batches):
            if batch_index != (self.n_batches-1):
                n_samples_per_batch = self.batch_size
            else:
                n_samples_per_batch = self.n_samples - (self.batch_size * (self.n_batches-1))
            batches[batch_index] = np.zeros((n_samples_per_batch, self.image_height, self.image_width, self.image_n_channels))
            batches_subtypes[batch_index] = [None] * n_samples_per_batch
        for batch_index in range(self.n_batches):
            print("reading batch " + str(batch_index) + "/" + str(self.n_batches-1))
            if batch_index != (self.n_batches-1):
                paths_of_images_of_batch = paths_of_images[(batch_index * self.batch_size) : ((batch_index+1) * self.batch_size)]
            else:
                paths_of_images_of_batch = paths_of_images[(batch_index * self.batch_size) :]
            for file_index, filename in enumerate(paths_of_images_of_batch):
                im = np.load(filename)
                batches[batch_index][file_index, :, :, :] = im
                batches_subtypes[batch_index][file_index] = filename.split("\\")[-2]
        return batches, batches_subtypes

    def read_batches_paths(self, path_dataset, path_save_test_patches):
        img_ext = '.npy'
        paths_of_images = [glob.glob(path_dataset+"\\**\\*"+img_ext)]
        paths_of_images = paths_of_images[0]
        # save paths of input data:
        if not os.path.exists(path_save_test_patches):
            os.makedirs(path_save_test_patches)
        with open(path_save_test_patches + 'paths_of_images.pickle', 'wb') as handle:
            pickle.dump(paths_of_images, handle)
        return paths_of_images

    def normalize_images(self, X_batch):
        # also see normalize_images() method in Utils.py
        X_batch = X_batch * (1. / 255) - 0.5
        return X_batch

    def load_network_model(self, saver_, session_, checkpoint_dir, model_dir_):
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

    def classify_with_1NN(self, embedding, labels, path_to_save):
        print("KNN on embedding data....")
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        neigh = KNeighborsClassifier(n_neighbors=2)   #--> it includes itself too
        neigh.fit(embedding, labels)
        y_pred = neigh.predict(embedding)
        accuracy_test = accuracy_score(y_true=labels, y_pred=y_pred)
        conf_matrix_test = confusion_matrix(y_true=labels, y_pred=y_pred)
        self.save_np_array_to_txt(variable=np.asarray(accuracy_test), name_of_variable="accuracy_test", path_to_save=path_to_save)
        self.save_variable(variable=accuracy_test, name_of_variable="accuracy_test", path_to_save=path_to_save)
        # self.plot_confusion_matrix(confusion_matrix=conf_matrix_test, class_names=[str(class_index+1) for class_index in range(n_classes)],
        #                            normalize=True, cmap="gray_r", path_to_save=path_to_save, name="test")
        self.plot_confusion_matrix(confusion_matrix=conf_matrix_test, class_names=[str(class_index) for class_index in range(n_classes)],
                                   normalize=True, cmap="gray_r", path_to_save=path_to_save, name="test")

    def plot_confusion_matrix(self, confusion_matrix, class_names, normalize=False, cmap="gray", path_to_save="./", name="temp"):
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix")
        else:
            pass
            # print('Confusion matrix, without normalization')
        # print(cm)
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        # plt.colorbar()
        tick_marks = np.arange(len(class_names))
        # plt.xticks(tick_marks, class_names, rotation=45)
        plt.xticks(tick_marks, class_names, rotation=0)
        plt.yticks(tick_marks, class_names)
        # tick_marks = np.arange(len(class_names) - 1)
        # plt.yticks(tick_marks, class_names[1:])
        fmt = '.2f' if normalize else 'd'
        thresh = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        plt.ylabel('true distortion type')
        plt.xlabel('predicted distortion type')
        n_classes = len(class_names)
        plt.ylim([n_classes - 0.5, -0.5])
        plt.tight_layout()
        # plt.show()
        plt.savefig(path_to_save + name + ".png")
        plt.clf()
        plt.close()

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(
                path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))
