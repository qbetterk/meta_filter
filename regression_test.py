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


train_num = 20     # number of training point in each domain
train_x   = [random.uniform(0, 2 * math.pi) for _ in range(train_num)]
train_y   = [[math.sin(i * d_sour_k[j] + d_sour_b[j]) for i in train_x] for j in range(d_sour_num)] 

support_num = 5
support_x   = [random.uniform(0, 2 * math.pi) for _ in range(support_num)]
support_y   = [[math.sin(i * d_sour_k[j] + d_sour_b[j]) for i in support_x] for j in range(d_targ_num)] 

test_num = 10
test_x   = [random.uniform(0, 2 * math.pi) for _ in range(test_num)]
test_y   = [[math.sin(i * d_sour_k[j] + d_sour_b[j]) for i in test_x] for j in range(d_targ_num)] 


# plt.plot(train_x, train_y[0], 'ro')
# plt.plot(train_x, train_y[1], 'bo')
# plt.show()

# print(d_sour_k)

# print(d_targ_k)

def sigmoid(x):
    if type(x) == np.ndarray:
        return np.array([sigmoid(i) for i in x])
    else:
        return 1 / (1 + math.exp(-1 * x))

def sigmoid_deri(x):
    if type(x) == np.ndarray:
        return np.array([sigmoid(i) * (1 - sigmoid(i)) for i in x])
    else:
        return sigmoid(x) * (1 - sigmoid(x))

class network:
    def __init__(self):
        self.hid_size = 16
        self.w1 = np.random.rand(1, self.hid_size)
        self.w2 = np.random.rand(self.hid_size, 1)

        self.learning_rate = 0.01

    def foward(self):
        self.hid_layer = sigmoid(np.dot(self.input, self.w1))
        self.output = np.dot(self.hid_layer, self.w2)
        self.loss = sum((self.output - self.oracle) ** 2)
        # print(self.loss)

    def backprop(self):
        tmp_deri = 2 * np.absolute(self.output - self.oracle) * sigmoid_deri(self.output)
        self.delta_w1 = np.dot(self.hid_layer.T, tmp_deri)
        self.delta_w2 = np.dot(self.input.T, np.dot(tmp_deri, self.w2.T) * sigmoid_deri(self.hid_layer))
        self.w1 -= self.learning_rate * self.delta_w1.T
        self.w2 -= self.learning_rate * self.delta_w2.T

    def train(self, x, y):
        self.input = np.array(x)
        self.oracle = np.array(y)
        self.foward()
        self.backprop()


    def test(self, x, y):
        self.input = np.array(x)
        self.oracle = np.array(y)
        self.hid_layer = sigmoid(np.dot(self.input, self.w1))
        self.output = sigmoid(np.dot(self.hid_layer, self.w2))
        self.loss = sum((self.output - self.oracle) ** 2)
        
        return [self.loss, self.output]


model = network()
for i in range(500):
    for j in range(train_num):
        model.train(train_x[j], train_y[0][j])

    if i %10 == 0:
        loss = sum([model.test(test_x[j], test_y[0][j])[0] for j in range(test_num)])
        print(loss)











