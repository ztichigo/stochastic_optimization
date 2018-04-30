
'''
# 随机梯度下降
# 《an overview of gradient descent optimization algorithms》
# URL：https://blog.csdn.net/google19890102/article/details/69942970
# URL: http://ruder.io/optimizing-gradient-descent/
'''
import logging
import numpy as np
from basemodel import BaseClassifier

class Adagrad(BaseClassifier):
    # def __init__(self):
    #     BaseClassifier.__init__()

    def fit(self, x_train, y_train, x_test, y_test):
        """Fit model in train data and calculate loss in both train data and test data

        Keyword Arguments:
            x_train: train data features
            y_train: train data labels
            x_test: test data features
            y_test: test data label
        """
        max_iter = 150000    # 最大迭代步数
        data_size = x_train.shape[0]
        dimension = 28*28
        w = (np.random.rand(dimension) - 0.5) * 0.04  # Initialize parameter.
        P = self.hypothesis(w, x_train)     # 预测概率
        loss_train = self.train_loss(w, y_train, P)   # 训练loss
        r = np.zeros(dimension)   # 用于记录梯度平方和
        
        f_train = open("./adagrad/adagrad_train_{}.txt".format(self.l1), "w", encoding="utf-8")
        f_test = open("./adagrad/adagrad_test_{}.txt".format(self.l1), "w", encoding="utf-8")
        count = 0  # 用于步长除以5
        count1 =0
        for k in range(max_iter+1):
            # pick a sample i uniformly at random
            i = np.random.randint(0, data_size)   # 不含最后一个
            g = self.grad(w, x_train[i], y_train[i])
            r = r + g * g
            w = w - self.init_stepsize * g / (np.sqrt(r) + 1e-8)

            if k % 1000 == 0:
                logging.info("Iter: {}".format(k))
                loss =loss_train 
                P = self.hypothesis(w, x_train)     # 预测概率
                loss_train = self.train_loss(w, y_train, P)   # 训练loss
                if loss_train - loss > 1e-6:
                    count +=1
                    
                P = self.hypothesis(w, x_test)     # 预测概率
                loss_test = self.test_loss(y_test, P)   # 测试loss
                
                logging.info("Loss in train data: {}".format(loss_train))
                logging.info("Loss in test data: {}".format(loss_test))

                f_train.write(str(loss_train))
                f_test.write(str(loss_test))
                f_train.write("\n")
                f_test.write("\n")
                print(self.init_stepsize)
            if count > 10 and count1 ==0:
                self.init_stepsize = self.init_stepsize/10
                count1 = 1
                

        f_train.close()
        f_test.close()
        self.w = w