'''
# 随机梯度下降
# 《an overview of gradient descent optimization algorithms》
# URL：https://blog.csdn.net/google19890102/article/details/69942970
# URL: http://ruder.io/optimizing-gradient-descent/
'''
import logging
import numpy as np
from basemodel import BaseClassifier
class Adam(BaseClassifier):

    def fit(self, x_train, y_train, x_test, y_test):
        """Fit model in train data and calculate loss in both train data and test data

        Keyword Arguments:
            x_train: train data features
            y_train: train data labels
            x_test: test data features
            y_test: test data label
        """
        max_iter = 150000
        data_size = x_train.shape[0]
        w = (np.random.rand(28 * 28) - 0.5) * 0.04
        P = self.hypothesis(w, x_train)     # 预测概率
        loss_train = self.train_loss(w, y_train, P)   # 训练loss
        r = np.zeros(28*28)   # 用于记录梯度平方和
        s = np.zeros(28*28)
        rho1 = 0.9
        rho2 = 0.99
        f_train = open("./adam/adam_train_{}.txt".format(self.l1), "w", encoding="utf-8")
        f_test = open("./adam/adam_test_{}.txt".format(self.l1), "w", encoding="utf-8")
        count = 0  # 用于步长除以10
        count1 =0
        for k in range(1, max_iter):
            # pick a sample i uniformly at random
            i = np.random.randint(0, data_size)
            g = self.grad(w, x_train[i], y_train[i])
            s = rho1 * s + (1 - rho1) * g
            r = rho2 * r + (1 - rho2) * g * g
            s_bar = s / (1 - rho1)
            r_bar = r / (1 - rho2)
            w = w - self.init_stepsize * s_bar / (np.sqrt(r_bar) + 1e-8)

            if k % 1000 == 0:
                print(self.init_stepsize)
                logging.info("Iter: {}".format(k))
                loss =loss_train 
                P = self.hypothesis(w, x_train)     # 预测概率
                loss_train = self.train_loss(w, y_train, P)   # 训练loss
                P = self.hypothesis(w, x_test)     # 预测概率
                loss_test = self.test_loss(y_test, P)   # 测试loss
                if loss_train - loss > 1e-6:
                    count +=1
                logging.info("Loss in train data: {}".format(loss_train))
                logging.info("Loss in test data: {}".format(loss_test))

                f_train.write(str(loss_train))
                f_test.write(str(loss_test))
                f_train.write("\n")
                f_test.write("\n")
            if count > 10 and count1 ==0:
                self.init_stepsize = self.init_stepsize/10
                count1 = 1

        f_train.close()
        f_test.close()
        self.w = w
        print ("iter:{}".format(k))