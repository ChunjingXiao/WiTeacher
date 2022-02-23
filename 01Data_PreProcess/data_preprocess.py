import os
import cv2
import matplotlib
import pandas as pd
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil
import h5py
from sklearn import preprocessing


def signfi_data2png(data_path, label_path, start_i, is_train, save_path):
    """
    signfi data processing
    """
    data = loadmat("./csi_data/signfi/" + data_path + ".mat")
    data = data[data_path]
    data = data.astype("float32")
    data = np.diff(data, n=1, axis=0, prepend=data[-1:, :, :, :])
    data = data.astype("uint8")
    label = loadmat("./csi_data/signfi/" + label_path + ".mat")
    label = label[label_path]
    labels = ""
    for i in range(0, np.size(label, 0)):
        png_data = data[:, :, :, i]
        png_label = label[i, 0]
        # Create file name
        file_name = str(i + start_i) + "_sign_" + str(png_label) + ".png"
        file_type = file_name + " sign_" + str(png_label) + "\n"
        if is_train:
            labels += file_type
        path = "./png_data/signfi/" + save_path + "/sign_" + str(
            png_label) + "/" + file_name
        mpimg.imsave(path, png_data)
    if is_train:
        fh = open('./png_data/signfi/00.txt', 'w', encoding='utf-8')
        fh.write(labels)
        fh.close()
    return start_i + np.size(label, 0)


def create_png_dir(out_path, a):
    """
    Create picture path
    :param out_path:
    :return:
    """
    for i in range(10):
        path1 = "./png_data/" + out_path + "/train/deep_" + str(i + 1)
        path2 = "./png_data/" + out_path + "/test/deep_" + str(i + 1)
        path3 = "./png_data/" + out_path + "/unlabel/deep_" + str(i + 1)

        isExist1s = os.path.exists(path1)
        if not isExist1s:
            os.makedirs(path1)

        isExist2s = os.path.exists(path2)
        if not isExist2s:
            os.makedirs(path2)
        isExist3s = os.path.exists(path3)
        if not isExist3s:
            os.makedirs(path3)


def deepSeg_data2png(out_path, data_path, label_path, is_test, start_i):
    """
    deepSeg data processing
    :param out_path:
    :param data_path:
    :param label_path:
    :param is_test:
    :param start_i:
    :return:
    """
    data = loadmat("./csi_data/deepseg/" + data_path + ".mat")
    data = data[data_path]

    create_png_dir(out_path)

    # Data preprocessing
    data = data.astype("float64")
    data = np.diff(data, axis=0, prepend=data[-1:, :, :, :])
    data = data.astype("uint8")
    if not is_test:
        savemat("./csi_data/deepseg/" + data_path + "_diff.mat", {data_path + "_diff": data})
    a = np.zeros(10, dtype=int)
    label = loadmat("./csi_data/deepseg/" + label_path + ".mat")
    label = label[label_path]
    unlabel_data = np.zeros((200, 30, 3, 150), dtype="uint8")
    labels = ""
    for i in range(0, np.size(label, 0)):
        png_data = data[:, :, :, i]

        # Create storage directory
        file_name = str(i + start_i) + "_our_" + str(label[i, 0]) + ".png"
        file_type = file_name + " our_" + str(label[i, 0]) + "\n"
        if not is_test:
            labels += file_type
        path1 = "./png_data/" + out_path + "/train/deep_" + str(
            label[i, 0]) + "/" + file_name
        path2 = "./png_data/" + out_path + "/test/deep_" + str(
            label[i, 0]) + "/" + file_name
        path3 = "./png_data/" + out_path + "/unlabel/deep_" + str(
            label[i, 0]) + "/" + file_name
        dist = 0
        if is_test:
            a[label[i, 0] - 1] += 1
            if (a[label[i, 0] - 1] <= 15):
                mpimg.imsave(path2, png_data)
            else:
                unlabel_data[:, :, :, dist] = png_data
                dist += 1
                mpimg.imsave(path3, png_data)
        else:
            mpimg.imsave(path1, png_data)
    if is_test:
        savemat("./csi_data/deepseg/csi_leave2_unlabel_diff.mat",
                {"csi_leave2_unlabel_diff": unlabel_data})

    if not is_test:
        fh = open('./png_data/' + out_path + '/00.txt', 'w', encoding='utf-8')
        fh.write(labels)
        fh.close()
    return start_i + np.size(label, 0)


if __name__ == '__main__':
    number = 0

    ## deepSeg data processing
    deepSeg_data_path = ["csi_leave2_test", "csi_leave2_train"]
    deepSeg_label_path = ["label_leave2_test", "label_leave2_train"]
    is_test = [True, False]
    for i in range():
        # number = deepSeg_data2png("leave_user2", deepSeg_data_path[i], deepSeg_label_path[i], is_test[i], 1)
        number = deepSeg_data2png("deepseg", deepSeg_data_path[i], deepSeg_label_path[i], is_test[i], 1)
    number = 0

    ## signfi data processing
    signfi_label_path = ["labelTest", "labelTrain", "labelUnlabel_all"]
    is_train = [False, True, False]
    save_path = ["test", "train", "unlabel"]
    signfi_data_path = ["csi_tensorTest", "csi_tensorTrain", "csi_tensorUnlabel_all"]
    for i in range(len(signfi_data_path)):
        number = signfi_data2png(signfi_data_path[i], signfi_label_path[i], number, is_train[i], save_path=save_path[i])
