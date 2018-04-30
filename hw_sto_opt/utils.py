#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 22:17:57 2018

@author: wangjinxin
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_data(opt="mnist"):

    """Load train and test dataset

    Keywords arguments:
        opt: dataset name: mnist or covertype

    Returns:
        (x_train, y_train): train data features and labels
        (x_test, y_test): test data features and labels
    """
    if opt == "mnist":
        train, test = tf.keras.datasets.mnist.load_data()
    
        x_train, y_train = train
        x_test, y_test = test
    
        x_train = x_train.reshape(x_train.shape[0], 28 * 28)
        x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    
        y_train = y_train.astype(np.int)
        y_test = y_test.astype(np.int)
        for i in range(len(y_train)):
            y_train[i] = 1 if y_train[i] % 2 == 0 else -1
        for i in range(len(y_test)):
            y_test[i] = 1 if y_test[i] % 2 == 0 else -1

    elif opt == "covertype":
        df = pd.read_csv("covtype.data", header=None)
        x = df.iloc[:, 0:54].values
        y = df[54].values
        for i in range(len(y)):
            y[i] = 1 if y[i] % 2 == 0 else -1
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        
    else:
        logging.error("Unknown dataset!!")

    logging.info("train data shape: {}".format(x_train.shape))
    logging.info("test data shape: {}".format(x_test.shape))
    return (x_train, y_train), (x_test, y_test)





