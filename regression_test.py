#!/usr/bin/env python3
#
import os, sys
import math, random
# import torch
# import torch.nn as nn
import pdb
import matplotlib.pyplot as plt
import numpy as np

random.seed(0)

d_sour_num = 5      # number of source domains
d_sour_k = [random.random() for _ in range(d_sour_num)]
d_sour_b = [random.uniform(0, 2 * math.pi) for _ in range(d_sour_num)]
d_targ_num = 1       # number of target domain
d_targ_k = [random.random() for _ in range(d_targ_num)]
d_targ_b = [random.uniform(0, 2 * math.pi) for _ in range(d_targ_num)]


train_num = 100     # number of training point in each domain
train_x   = [random.uniform(0, 2 * math.pi) for _ in range(train_num)]
train_y   = [[math.sin(i * d_sour_k[j] + d_sour_b[j]) for i in train_x] for j in range(d_sour_num)]

val_num = 100
val_x   = [random.uniform(0, 2 * math.pi) for _ in range(val_num)]
val_y   = [[math.sin(i * d_sour_k[j] + d_sour_b[j]) for i in val_x] for j in range(d_sour_num)] 

test_num = 100
test_x   = [random.uniform(0, 2 * math.pi) for _ in range(test_num)]
test_y   = [[math.sin(i * d_sour_k[j] + d_sour_b[j]) for i in test_x] for j in range(d_sour_num)] 

support_num = 5
support_x   = [random.uniform(0, 2 * math.pi) for _ in range(support_num)]
support_y   = [[math.sin(i * d_targ_k[j] + d_targ_b[j]) for i in support_x] for j in range(d_targ_num)] 

# test_num = 10
# test_x   = [random.uniform(0, 2 * math.pi) for _ in range(test_num)]
# test_y   = [[math.sin(i * d_targ_k[j] + d_targ_b[j]) for i in test_x] for j in range(d_targ_num)] 


# plt.plot(train_x, train_y[0], 'ro')
# plt.plot(train_x, train_y[1], 'bo')
# plt.show()

# print(d_sour_k)

# print(d_targ_k)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deri(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(x, 0)

def relu_deri(x):
    x[x >= 0] = 1
    x[x == 0] = 0
    return x

class network:
    def __init__(self):
        self.hid_size = [40, 40]
        self.w1 = np.random.rand(self.hid_size[0], 1)
        self.b1 = np.zeros((self.hid_size[0], 1))
        self.w2 = np.random.rand(self.hid_size[1], self.hid_size[0])
        self.b2 = np.zeros((self.hid_size[1], 1))
        self.w3 = np.random.rand(1, self.hid_size[1])
        self.b3 = np.zeros((1, 1))

        self.learning_rate = 0.001

    def foward(self):
        # pdb.set_trace()
        self.hid_layer1 = sigmoid(np.dot(self.w1, self.input) + self.b1)
        self.hid_layer2 = sigmoid(np.dot(self.w2, self.hid_layer1) + self.b2)
        self.output     =      np.dot(self.w3, self.hid_layer2) + self.b3

    def backprop(self):
        # pdb.set_trace()
        y_deri = 2 * (self.output - self.oracle) 
        self.delta_b3 = np.expand_dims(np.sum(y_deri, axis = 1), axis = 1)
        self.delta_w3 = np.dot(y_deri, self.hid_layer2.T)

        h2_deri = np.dot(self.w3.T, y_deri)
        # self.delta_b2 = np.expand_dims(np.sum(h2_deri * sigmoid_deri(self.hid_layer2), axis = 1), axis = 1)
        self.delta_b2 = (np.sum(h2_deri * sigmoid_deri(self.hid_layer2), axis = 1)).reshape(-1, 1)
        self.delta_w2 = np.dot(h2_deri * sigmoid_deri(self.hid_layer2), self.hid_layer1.T)

        h1_deri = np.dot(self.w2.T, h2_deri * sigmoid_deri(self.hid_layer2))
        # self.delta_b1 = np.expand_dims(np.sum(h1_deri * sigmoid_deri(self.hid_layer1), axis = 1), axis = 1)
        self.delta_b1 = (np.sum(h1_deri * sigmoid_deri(self.hid_layer1), axis = 1)).reshape(-1, 1)
        self.delta_w1 = np.dot(h1_deri * sigmoid_deri(self.hid_layer1), self.input.T)

        self.w1 -= self.learning_rate * self.delta_w1
        self.w2 -= self.learning_rate * self.delta_w2
        self.w3 -= self.learning_rate * self.delta_w3
        self.b1 -= self.learning_rate * self.delta_b1
        self.b2 -= self.learning_rate * self.delta_b2
        self.b3 -= self.learning_rate * self.delta_b3

    def train(self, x, y):
        self.input = np.array([x])
        self.oracle = np.array(y)
        self.foward()
        self.backprop()


    def test(self, x, y=None):
        self.input = np.array([x])
        # self.hid_layer = sigmoid(np.dot(self.w1, self.input))
        # self.output = np.dot(self.w2, self.hid_layer)
        self.hid_layer1 = sigmoid(np.dot(self.w1, self.input) + self.b1)
        self.hid_layer2 = sigmoid(np.dot(self.w2, self.hid_layer1) + self.b2)
        self.output     =      np.dot(self.w3, self.hid_layer2) + self.b3
        if y: 
            self.oracle = np.array(y)
            self.loss = np.sum((self.output - self.oracle) ** 2)
            return self.loss, self.output
        return self.output

def main():
    model = network()
    pre_val_loss = 99999
    for i in range(100000):
        # for j in range(train_num):
        #     model.train(train_x[j:j+1], train_y[0][j:j+1])
        for j in range(int(train_num/30)):
            model.train(train_x[30*j:30*(j+1)], train_y[0][30*j:30*(j+1)])


        # model.train(train_x, train_y[0])


        # val_loss = model.test(val_x, val_y[0])[0]
        # if type(val_loss) != np.float64:
        #     print(val_loss)
        #     pdb.set_trace()
        # print(val_loss)
        # # pdb.set_trace()

        if i %100 == 0:
            val_loss = model.test(val_x, val_y[0])[0]
            train_loss = model.test(train_x, train_y[0])[0]
            # if val_loss == 'nan':
            #     pdb.set_trace()
            # if val_loss < 0.12:
            #     break

            print(i, val_loss, train_loss)

            # if val_loss < 0.1:
            #     break
            # if val_loss < pre_val_loss:
            #     model_optim = model
            #     pre_val_loss = val_loss
            # print(pre_val_loss)


    test_loss = model.test(test_x, test_y[0])[0]
    # test_loss = model_optim.test(test_x, test_y[0])[0]
    print(test_loss)

    # pdb.set_trace()
    # plt.plot(train_x, model.test(train_x)[0], 'bo')
    # plt.plot(train_x, train_y[0], 'ro')

    # plt.plot(val_x, model.test(val_x)[0],'bo')
    # plt.plot(val_x, val_y[0],'ro')

    plt.plot(test_x, model.test(test_x)[0],'bo')
    plt.plot(test_x, [math.sin(i * d_sour_k[0] + d_sour_b[0]) for i in test_x],'ro')
    plt.show()


if __name__ == "__main__":
    main()









