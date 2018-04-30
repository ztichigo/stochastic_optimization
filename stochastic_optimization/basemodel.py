
import logging
import numpy as np
import tensorflow as tf

class BaseClassifier(object):
    """Base Class of other algorithms, such as adagrad, adam, rmsprop, SVRG, FTRL and so on.
    This's very useful  """
    def __init__(self, init_stepsize=0.001, l1=0., l2=0.001):
        self.init_stepsize = init_stepsize  # 步长
        self.l1 = l1        # 一范数正则化系数
        self.l2 = l2        # 二范数正则化系数
        self.data_size = None
        self.w = None

    def hypothesis(self, w, X):
        """Prediction function

        Keyword arguments:
            w: logistic regression parameter
            X: all train/test samples
        Returns:
            probability value that X belongs to label 1
        """
        if X.ndim == 2:
            data_size = X.shape[0]    # data_size
        P = np.zeros(data_size)   
        for i in range(data_size):
            wTx = np.dot(X[i], w)
            prob = 1. / (1. + np.exp(-max(min(wTx, 350.), -350.)))   # 判断为label1的概率
            prob = max(min(prob, 1. - 1e-15), 1e-15)
            P[i]=prob
        return P
    
    def train_loss(self, w, Y, P):
        data_size = Y.shape[0]
        loss = 0
        for i in range(data_size):
            #print("P ",P[i])
            logloss = -np.log(P[i])  if Y[i] == 1 else -np.log(1. - P[i])
            loss += logloss
        loss /= data_size
        loss += self.l1 * np.linalg.norm(w, 1) + self.l2 * np.dot(w,w)
        return loss
    
    def test_loss(self, Y, P):
        d = Y.shape[0]
        loss = 0
        for i in range(d):
            logloss = -np.log(P[i])  if Y[i] == 1 else -np.log(1. - P[i])
            loss += logloss
        loss /= d
        return loss

    def grad(self, w, X, Y):
        """gradient with parameter w """

        def grad_for_one(x, y):    
            """gradient for one sample
               if 单个样本，remember to divide it by data_size
            """
            # subgradient of l1 penalty
            subgrad = np.zeros([w.shape[0], ])
            subgrad[w > 0.0] = 1
            subgrad[w < 0.0] = -1
            # return gradient of loss
            return (1./ (1 + np.exp(max(min(np.dot(y * w, x), 350.), -350.)))  * (-y * x) + self.l1 * subgrad + 2 * self.l2 * w)
        if X.ndim == 1:
            return grad_for_one(X, Y)
        # more than one samples
        elif X.ndim == 2:
            grad = 0
            for i in range(X.shape[0]):
                grad += grad_for_one(X[i], Y[i])
            grad /= X.shape[0]
            return grad
        
    def predict(self, w, x_test):
        """Predict in x_test"""
        P = self.hypothesis(w, x_test)

        P[P >= 0.5] = 1
        P[P < 0.5] = -1

        return P     # 预测

