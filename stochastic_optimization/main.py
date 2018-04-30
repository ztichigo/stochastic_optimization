import logging
from sklearn import metrics
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse         # 处理稀疏矩阵
from scipy.sparse import csr_matrix
from math import exp, log, sqrt
import random

from utils import load_data
from ftrl import ftrl_proximal
#from svrg import SVRG
#from svrg_bb import SVRG_BB
#from adagrad import Adagrad
#from adam import Adam


#def main():
#    logging.basicConfig(level=logging.INFO)
#    
#    # model = SVRG(init_stepsize=0.000002, l1=0.001, l2=0.)
#    # model = SVRG_BB(init_stepsize=0.000002, l1=0.1, l2=0.)
#    #model = Adagrad(init_stepsize=0.003, l1=0.001, l2=0.)
#    # adagrad 步长选取  0.003
#    # model = Adam(init_stepsize=0.0001, l1=0.001, l2=0.)
#    
#
#    (x_train, y_train), (x_test, y_test) = load_data(opt="mnist")
#    model.fit(x_train, y_train, x_test, y_test)
#    y_pred = model.predict(model.w ,x_test)
#
#    # measure
#    f1_score = metrics.f1_score(y_test, y_pred, average="binary")
#    accuracy = metrics.accuracy_score(y_test, y_pred)
#    logging.info("f1-score: {}".format(f1_score))
#    logging.info("accuracy: {}".format(accuracy))
#
if __name__ == "__main__":
    # main()

    (x_train, y_train),(x_test, y_test) = load_data()
    # parameters 
    alpha = .01  # learning rate
    beta = 1.   # smoothing parameter for adaptive learning rate
    L1 = 0.001     # L1 regularization, larger value means more regularized
    L2 = 0.     # L2 regularization, larger value means more regularized
    D = 28*28             # number of weights to use
    learner = ftrl_proximal(alpha, beta, L1, L2, D)    
    w = np.random.randn(784).reshape(784,1)   # 随机初始化w
    EPOCH = 20  
    loss = 0
    count = 0
    start = datetime.now()
    wa = []
    wb = []
    for epoch in range(1, EPOCH):  
    
        for i in range(x_train.shape[0]):
            p = learner.predict(x_train[i])
            loss += learner.log_loss(p, y_train[i])
            count += 1
            learner.update(x_train[i], p, y_train[i])
            acc = 0
            if count%10000==0:
                wa.append(loss/count)
                w =  learner.w
                test_loss = 0
                for i  in range(x_test.shape[0]):
                    p1 = learner.predict(x_test[i])
                    test_loss += learner.log_loss(p1, y_test[i])
                    if p1 >0.5:
                        pred =1
                    else:
                        pred = -1
                    if y_test[i] == pred:
                        acc +=1
           
                wb.append(test_loss/10000)
                print('Epoch {} | count {} | eclipsed time: {} | loss: {} |test_loss:{} | test_acc: {}/{}'.format(epoch, count, datetime.now()-start, loss/count ,test_loss/10000, acc , 10000))
