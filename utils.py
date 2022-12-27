import os
import numpy as np
import tensorflow as tf
import h5py
import math
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from matplotlib.pyplot import imshow

def load_dataset():
    train_dataset = h5py.File('data/train_signs.h5', "r")
    # your train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])

    '''
    img = Image.fromarray(train_set_x_orig[im])
    for im in range(len(train_set_x_orig)):
        img = Image.fromarray(train_set_x_orig[im])
        #img = ImageEnhance.Contrast(img).enhance(1)
        img = ImageOps.equalize(img, mask = None)
        img = img.filter(ImageFilter.SMOOTH)
        train_set_x_orig[im] = np.array(img)
    train_set_x_orig2 = []
    for im in range(len(train_set_x_orig)):
        train_set_x_orig2.append(np.rot90(train_set_x_orig[im], axes=(-3, -2)))
    train_set_x_orig3 = []
    for im in range(len(train_set_x_orig)):
        train_set_x_orig3.append(np.rot90(train_set_x_orig2[im], axes=(-3, -2)))
    train_set_x_orig3 = np.array(train_set_x_orig3)
    train_set_x_orig4 = []
    for im in range(len(train_set_x_orig)):
        train_set_x_orig4.append(np.rot90(train_set_x_orig3[im], axes=(-3, -2)))
    train_set_x_orig4 = np.array(train_set_x_orig4)

    train_set_x_orig = np.concatenate((train_set_x_orig, train_set_x_orig2, train_set_x_orig3, train_set_x_orig4))
    '''
    
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    #train_set_y_orig = np.concatenate((train_set_y_orig, train_set_y_orig, train_set_y_orig, train_set_y_orig))

    
    test_dataset = h5py.File('data/test_signs.h5', "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y