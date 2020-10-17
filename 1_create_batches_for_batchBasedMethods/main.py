
import numpy as np
# import pandas as pd
from random import shuffle
import random
# from skimage.morphology import binary_erosion
import time
from PIL import Image
# import PIL
import glob
import matplotlib as plt
import os, os.path
import pandas as pd
import tensorflow as tf
import pickle


def main():
    set_name = "test1"
    # batch_size = 16 * 3
    batch_size = 9 * 5
    n_samples_total = 15030  # divisable by 9
    tissue_type_list = ["00_TUMOR", "01_STROMA", "02_MUCUS", "03_LYMPHO", "04_DEBRIS", "05_SMOOTH_MUSCLE", "06_ADIPOSE", "07_BACKGROUND", "08_NORMAL"]
    path_base = "D:\\Datasets\\CRC_new_large\\CRC_100K_train_test_numpy\\" + set_name + "\\"
    path_save_batches = ".\\batches\\"
    batches = make_batches_RANDOM(tissue_type_list, batch_size, n_samples_total, path_base, path_save_batches)
    
def make_batches_RANDOM(tissue_type_list, batch_size, n_samples_total, path_base, path_save_batches):
    n_batches = int(np.ceil((n_samples_total * 1.0) / batch_size))
    n_classes = len(tissue_type_list)
    n_ClassSamples_per_batch = int(np.ceil((batch_size * 1.0) / n_classes))
    batches = [None] * n_batches
    for batch_index in range(n_batches):
        batches[batch_index] = []
    filename_classes = [None] * n_classes
    for class_index, class_name in enumerate(tissue_type_list):
        path_class = path_base + class_name + "\\"
        path_class_list = glob.glob(path_class + "*.npy")
        filename_class = [path_class_list[i].split("\\")[-1] for i in range(len(path_class_list))]
        shuffle(filename_class)
        filename_classes[class_index] = filename_class
    for batch_index in range(n_batches):
        for class_index, filename_class in enumerate(filename_classes):
            filename_class_batch = random.sample(filename_class, n_ClassSamples_per_batch)
            batches[batch_index].extend(filename_class_batch)
    if not os.path.exists(path_save_batches):
        os.makedirs(path_save_batches)
    with open(path_save_batches + 'batches.pickle', 'wb') as handle:
        pickle.dump(batches, handle)
    return batches

def make_batches_NOT_RANDOM(tissue_type_list, batch_size, n_samples_total, path_base, path_save_batches):
    n_batches = int(np.ceil((n_samples_total * 1.0) / batch_size))
    n_classes = len(tissue_type_list)
    n_ClassSamples_per_batch = int(np.ceil((batch_size * 1.0) / n_classes))
    batches = [None] * n_batches
    for batch_index in range(n_batches):
        batches[batch_index] = []
    filename_classes = [None] * n_classes
    for class_index, class_name in enumerate(tissue_type_list):
        path_class = path_base + class_name + "\\"
        path_class_list = glob.glob(path_class + "*.npy")
        filename_class = [path_class_list[i].split("\\")[-1] for i in range(len(path_class_list))]
        shuffle(filename_class)
        n_samples_per_class = len(filename_class)
        filename_classes[class_index] = filename_class
    for batch_index in range(n_batches):
        for class_index, filename_class in enumerate(filename_classes):
            start_index = min((batch_index*n_ClassSamples_per_batch), n_samples_per_class)
            end_index = min((batch_index+1)*n_ClassSamples_per_batch, n_samples_per_class)
            filename_class_batch = filename_class[start_index:end_index]
            batches[batch_index].extend(filename_class_batch)
        pass
    pass


if __name__ == "__main__":
    main()