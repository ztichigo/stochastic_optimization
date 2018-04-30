import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    x = np.arange(3000,150000-2000, 1000)
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    with open("./adam/adam_train_0.001.txt", "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            y1.append(float(line))
    with open("./adam/adam_test_0.001.txt", "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            y2.append(float(line))
    
#    with open("adam_test_0.001.txt", "r", encoding="utf-8") as fr:
#        for line in fr.readlines():
#            y1.append(float(line))
#
#    with open("adam_test_0.1.txt", "r", encoding="utf-8") as fr:
#        for line in fr.readlines():
#            y2.append(float(line))
#
#    with open("adam_test_1.txt", "r", encoding="utf-8") as fr:
#        for line in fr.readlines():
#            y3.append(float(line))
#
#    with open("adam_test_10.txt", "r", encoding="utf-8") as fr:
#        for line in fr.readlines():
#            y4.append(float(line))

    plt.plot(x, y1[3:1499], "r", label="lambda=0.001")
#    plt.plot(x, y2[0:100], "b", label="lambda=0.1")
#    plt.plot(x, y3[0:100], "g", label="lambda=1")
#    plt.plot(x, y4[0:100], "y", label="lambda=10")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.title("adam")
    plt.legend()
    plt.savefig("./adam/adam_train1.jpg")