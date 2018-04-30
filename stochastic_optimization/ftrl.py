#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 22:26:55 2018

@author: wangjinxin
FTRLP实现
# ref: https://www.kaggle.com/c/avazu-ctr-prediction/discussion/10927
# ref: https://www.kaggle.com/jiweiliu/ftrl-starter-code/code
# https://github.com/fmfn/FTRLp/blob/master/FTRLp.py 
# 分布式实现： https://github.com/Tencent/angel/blob/master/docs/algo/ftrl_lr_spark.md
# 相关论文
# 1. 《Online Learning and Online
# Convex Optimization1》section 2.2,2.3 URL: url{http://www.cs.huji.ac.il/~shais/papers/OLsurvey.pdf}
# 3.\ 原始论文：\\
# URL: https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
# 2. \  from 美团点评技术团队
# URL: https://tech.meituan.com/online-learning.html
# 4.\ 其他：\\
# URL: http://www.datakit.cn/blog/2016/05/11/ftrl.html
"""

from datetime import datetime
from math import exp, log, sqrt
import numpy as np
# parameters 
alpha = .1  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 0.1     # L1 regularization, larger value means more regularized
L2 = 0.     # L2 regularization, larger value means more regularized
D = 784             # number of weights to use

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal
        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization
    '''

    def __init__(self, alpha, beta, L1, L2, D):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = np.zeros(D)
        self.z = np.zeros(D)
        self.w = np.zeros(D)     ## 可以考虑改进, 因为x是稀疏的

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        D = self.D
        n = self.n
        z = self.z
        w = self.w

        # wTx is the inner product of w and x
        wTx = 0.
        for i in range(D):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)
        wTx = np.dot(x,w)      
        # cache the current w for update stage
        self.w = w
        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w
        D = self.D

        # update z and n
        if y ==-1:
            for i in range(D):
                g = p * x[i]     # gradient 计算
                sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
                z[i] += g - sigma * w[i]
                n[i] += g * g
        else:
            for i in range(D):
                g = (p-1) * x[i]     # gradient 计算
                sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
                z[i] += g - sigma * w[i]
                n[i] += g * g
            
            


    def log_loss(self, p, y):
        ''' FUNCTION: Bounded logloss
    
            INPUT:
                p: our prediction
                y: real answer
    
            OUTPUT:
                logarithmic loss of p given y
        '''
    
        p = max(min(p, 1. - 10e-15), 10e-15)
        return -log(p) if y == 1. else -log(1. - p)

