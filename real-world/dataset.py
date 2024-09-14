# -*- coding: utf-8 -*-

import numpy as np
import os
import pdb


data_dir = "./data"

def load_data(name="coat"):

    if name == "coat":
        data_set_dir = os.path.join(data_dir, name)
        train_file = os.path.join(data_set_dir, "train.ascii")
        test_file = os.path.join(data_set_dir, "test.ascii")

        with open(train_file, "r") as f:
            x_train = []
            for line in f.readlines():
                x_train.append(line.split())

            x_train = np.array(x_train).astype(int)

        with open(test_file, "r") as f:
            x_test = []
            for line in f.readlines():
                x_test.append(line.split())

            x_test = np.array(x_test).astype(int)

        print("===>Load from {} data set<===".format(name))
        print("[train] rating ratio: {:.6f}".format((x_train>0).sum() / (x_train.shape[0] * x_train.shape[1])))
        print("[test]  rating ratio: {:.6f}".format((x_test>0).sum() / (x_test.shape[0] * x_test.shape[1])))

    elif name == "yahoo":
        data_set_dir = os.path.join(data_dir, name)
        train_file = os.path.join(data_set_dir,
            "ydata-ymusic-rating-study-v1_0-train.txt")
        test_file = os.path.join(data_set_dir,
            "ydata-ymusic-rating-study-v1_0-test.txt")

        x_train = []
        # <user_id> <song id> <rating>
        with open(train_file, "r") as f:
            for line in f:
                x_train.append(line.strip().split())
        x_train = np.array(x_train).astype(int)

        x_test = []
        # <user_id> <song id> <rating>
        with open(test_file, "r") as f:
            for line in f:
                x_test.append(line.strip().split())
        x_test = np.array(x_test).astype(int)
        print("===>Load from {} data set<===".format(name))
        print("[train] num data:", x_train.shape[0])
        print("[test]  num data:", x_test.shape[0])

        return x_train[:,:-1], x_train[:,-1], \
            x_test[:, :-1], x_test[:,-1]

    else:
        print("Cant find the data set",name)
        return

    return x_train, x_test


def load_features(name="coat"):

    if name == "coat":
        data_set_dir = os.path.join(data_dir, name, 'user_item_features')
        user_feature_file = os.path.join(data_set_dir, "user_features.ascii")
        item_feature_file = os.path.join(data_set_dir, "item_features.ascii")

        with open(user_feature_file, "r") as f:
            x_user = []
            for line in f.readlines():
                x_user.append(line.split())

            x_user = np.array(x_user).astype(int)

        with open(item_feature_file, "r") as f:
            x_item = []
            for line in f.readlines():
                x_item.append(line.split())

            x_item = np.array(x_item).astype(int)

        print("===>Load from {} data set<===".format(name))

    else:
        print("Cant find the data set",name)
        return

    return x_user, x_item


def rating_mat_to_sample(mat):
    row, col = np.nonzero(mat)
    y = mat[row,col]
    x = np.concatenate([row.reshape(-1,1), col.reshape(-1,1)], axis=1)
    return x, y