#!/usr/bin/env python3

import os
import io

import pickle
import argparse

import numpy as np


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit


seed = 7
test_size = 0.3

np.random.seed(seed)

def load_data(input_file):

    Y = []

    #load the dic of activities
    if "MILAN" in path:
        pickle_in = open("../datasets/milan_activity_list.pickle","rb")

    if "ARUBA" in path:
        pickle_in = open("../datasets/aruba_activity_list.pickle","rb")


    dictActivities = pickle.load(pickle_in)

    # get all keys to an array
    *listActivities, = dictActivities
    

    name_x = "{}_x.npy".format(input_file)
    name_y = "{}_y.npy".format(input_file)

    X = np.load(name_x,allow_pickle=True)

    Y = np.load(name_y,allow_pickle=True)

    # tokenize Y
    Y1 = Y
    for i,y in enumerate(Y):
        Y1[i]=dictActivities[y]

    Y=np.asarray(Y1)

    return X, Y, dictActivities, listActivities


if __name__ == '__main__':

    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--i', dest='input', action='store', default='', help='input')
    p.add_argument('--p', dest='path', action='store', default='', help='path')

    args = p.parse_args()


    input_file = str(args.input)
    path = str(args.path)


    #if the folder doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    X, Y, dictActivitie, listActivities = load_data(input_file)

    print("TOTAL")
    print(X.shape)
    print(Y.shape)

    # Split the dataset into Train and Test
    stratShufleSplit = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_index, test_index in stratShufleSplit.split(X,Y):
        data_X_train, data_X_test = X[train_index], X[test_index]
        data_Y_train, data_Y_test = Y[train_index], Y[test_index]


    np.save("{}/{}_X_TEST".format(path,input_file), data_X_test)
    np.save("{}/{}_Y_TEST".format(path,input_file), data_Y_test)

    print("\nTEST")
    print("X_TEST : {}".format(data_X_test.shape))
    print("Y_TEST : {}".format(data_Y_test.shape))


    # Cross Validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
    k = 0

    for k_train_index, k_validation_index in kfold.split(data_X_train, data_Y_train, groups=None):

        np.save("{}/{}_X_TRAIN_{}".format(path,input_file,k), data_X_train[k_train_index])
        np.save("{}/{}_Y_TRAIN_{}".format(path,input_file,k), data_Y_train[k_train_index])

        np.save("{}/{}_X_VALIDATION_{}".format(path,input_file,k), data_X_train[k_validation_index])
        np.save("{}/{}_Y_VALIDATION_{}".format(path,input_file,k), data_Y_train[k_validation_index])

        print("\nTrain / Validation subset n°{}".format(k+1))

        print("TRAIN")
        print("X_TRAIN_{} : {}".format(k+1,data_X_train[k_train_index].shape))
        print("Y_TRAIN_{} : {}".format(k+1,data_Y_train[k_train_index].shape))

        print("VALIDATION")
        print("X_VALIDATION_{} : {}".format(k+1,data_X_train[k_validation_index].shape))
        print("X_VALIDATION_{} : {}".format(k+1,data_Y_train[k_validation_index].shape))
        
        k += 1
