#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 22:27:34 2018

@author: wangjinxin
SVRG-BB 算法
ref: 《Barzilai-Borwein Step Size for Stochastic Gradient Descent》NIPS2016
"""
# 先做几步sgd， datetime
import logging
import numpy as np
from basemodel import BaseClassifier

class SVRG_BB(BaseClassifier):

    def fit(self, x_train, y_train, x_test, y_test):
        """Fit model with train features and labels."""
        outiter = 10    # 最大迭代步数
        m = 60000       # 内层循环
        data_size = x_train.shape[0]
        w = np.random.randn(28 * 28)/20  # Initialize parameter.
        
        f_train = open("svrg_bb_train_{}.txt".format(self.l1), "w", encoding="utf-8")
        f_test = open("svrg_bb_test_{}.txt".format(self.l1), "w", encoding="utf-8")
        alpha = self.init_stepsize
        for k in range(16):    # 先做几步SGD
            i = np.random.randint(0, data_size)
            w -= self.init_stepsize/10 * self.grad(w, x_train[i], y_train[i])

        for e in range(1, outiter+1):
            logging.info("epoch: {}".format(e))
            if e ==1:
                w0 = w            # w0代表外层循环的变量，上层变量
                fg0 = self.grad(w0, x_train, y_train)    # 计算上次全梯度
            else:
                # print("neiji: ",np.linalg.norm(w1-w0, 2), np.linalg.norm(fg1-fg0,2))
                alpha = np.linalg.norm(w1-w0)**2 / np.dot(w1-w0, fg1-fg0) / m
                # print("alpha: ", alpha)
                w0 = w            # w0代表外层循环的变量，上层变量
                fg0 = self.grad(w0, x_train, y_train)    # 计算上次全梯度
            # wa = []
            for j in range(m):                  # 内循环
                # pick a sample i uniformly at random
                i = np.random.randint(0, data_size)
                v = self.grad(w, x_train[i], y_train[i]) - self.grad(w0, x_train[i], y_train[i]) + fg0
                w = w - alpha * v
                if j % 10000 == 0:
                    logging.info("Iter: {}".format(j))
                    P = self.hypothesis(w, x_train)     # 预测概率
                    loss_train = self.train_loss(w, y_train, P)   # 训练loss
                        
                    P = self.hypothesis(w, x_test)     # 预测概率
                    loss_test = self.test_loss(y_test, P)   # 测试loss
                    
                    logging.info("Loss in train data: {}".format(loss_train))
                    logging.info("Loss in test data: {}".format(loss_test))
    
                    f_train.write(str(loss_train))
                    f_test.write(str(loss_test))
                    f_train.write("\n")
                    f_test.write("\n")
            w1 = w    # 本次变量
            fg1 = self.grad(w1, x_train, y_train)    # 计算本次全梯度

        f_train.close()
        f_test.close()
        self.w = w
