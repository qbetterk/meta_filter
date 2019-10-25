#!/usr/bin/env python3
#
import os, sys
import math, random
import numpy as np
import copy

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

import pdb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle

random.seed(0)
torch.manual_seed(1)

d_sour_num = 20      # number of source domains
d_sour_a = [random.uniform(0.1, 5.0) for _ in range(d_sour_num)]
d_sour_b = [random.uniform(0, math.pi) for _ in range(d_sour_num)]
d_targ_num = 1       # number of target domain
d_targ_a = [random.uniform(0.1, 5.0) for _ in range(d_targ_num)]
d_targ_b = [random.uniform(0, math.pi) for _ in range(d_targ_num)]


train_num = 10     # number of training point in each domain
train_x   = np.array([random.uniform(-5.0, 5.0) for _ in range(train_num)], dtype=np.float32).reshape(-1,1)
train_y   = np.array([[d_sour_a[j] * math.sin(i + d_sour_b[j]) for i in train_x] for j in range(d_sour_num)], dtype=np.float32).reshape(d_sour_num, train_num, 1)

val_num = 100
val_x   = np.array([random.uniform(-5.0, 5.0) for _ in range(val_num)], dtype=np.float32).reshape(-1,1)
val_y   = np.array([[d_sour_a[j] * math.sin(i + d_sour_b[j]) for i in val_x] for j in range(d_sour_num)], dtype=np.float32).reshape(d_sour_num, val_num, 1)

support_num = 5
support_x   = np.array([random.uniform(-5.0, 5.0) for _ in range(support_num)], dtype=np.float32).reshape(-1,1)
support_y   = np.array([[d_targ_a[j] * math.sin(i + d_targ_b[j]) for i in support_x] for j in range(d_targ_num)], dtype=np.float32).reshape(d_targ_num, support_num, 1)

test_num = 100
test_x   = np.array([random.uniform(-5.0, 5.0) for _ in range(test_num)], dtype=np.float32).reshape(-1,1)
test_y   = np.array([[d_targ_a[j] * math.sin(i + d_targ_b[j]) for i in test_x] for j in range(d_targ_num)], dtype=np.float32).reshape(d_targ_num, test_num, 1)

test_x_old   = np.array([random.uniform(-5.0, 5.0) for _ in range(test_num)], dtype=np.float32).reshape(-1,1)
test_y_old   = np.array([[d_sour_a[j] * math.sin(i + d_sour_b[j]) for i in test_x_old] for j in range(d_sour_num)], dtype=np.float32).reshape(d_sour_num, test_num, 1)

# pdb.set_trace()



class Network(nn.Module):
    def __init__(self, input_size=1, output_size=1, learning_rate=0.01):
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hid_size = [40, 40]

        self.linear1  = nn.Linear(self.input_size, self.hid_size[0])
        self.linear2  = nn.Linear(self.hid_size[0], self.hid_size[1])
        self.linear3  = nn.Linear(self.hid_size[1], self.output_size)

    def forward(self, inputs):
        self.input = inputs
        self.hid_layer1 = F.relu(self.linear1(self.input))
        self.hid_layer2 = F.relu(self.linear2(self.hid_layer1))
        self.output = self.linear3(self.hid_layer2)
        return self.output

    def backward(self):
        with torch.no_grad():
            # self.linear1.weight -= self.learning_rate * self.linear1.weight.grad
            # self.linear1.bias -= self.learning_rate * self.linear1.bias.grad
            # self.linear2.weight -= self.learning_rate * self.linear2.weight.grad
            # self.linear2.bias -= self.learning_rate * self.linear2.bias.grad
            # self.linear3.weight -= self.learning_rate * self.linear3.weight.grad
            # self.linear3.bias -= self.learning_rate * self.linear3.bias.grad

            self.linear1.weight.data = self.linear1.weight.data - self.learning_rate * self.linear1.weight.grad
            self.linear1.bias.data = self.linear1.bias.data - self.learning_rate * self.linear1.bias.grad
            self.linear2.weight.data = self.linear2.weight.data - self.learning_rate * self.linear2.weight.grad
            self.linear2.bias.data = self.linear2.bias.data - self.learning_rate * self.linear2.bias.grad
            self.linear3.weight.data = self.linear3.weight.data - self.learning_rate * self.linear3.weight.grad
            self.linear3.bias.data = self.linear3.bias.data - self.learning_rate * self.linear3.bias.grad

class Filter(Network):
    def __init__(self):
        super(Filter, self).__init__()

    def backward():
        pass


def numpy_to_var(batch, batch_size, last_batch=True, domain=0, **kwargs):
    if 'x' in kwargs.keys():
        x = kwargs['x']
        if last_batch:
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(x[batch * batch_size:]).cuda())
            else:
                inputs = Variable(torch.from_numpy(x[batch * batch_size:]))
        else:
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(x[batch * batch_size: (batch + 1) * batch_size]).cuda())
            else:
                inputs = Variable(torch.from_numpy(x[batch * batch_size: (batch + 1) * batch_size]))

    if 'y' in kwargs.keys():
        y = kwargs['y']
        if last_batch:
            if torch.cuda.is_available():
                labels = Variable(torch.from_numpy(y[domain][batch * batch_size:]).cuda())
            else:
                labels = Variable(torch.from_numpy(y[domain][batch * batch_size:]))
        else:
            if torch.cuda.is_available():
                labels = Variable(torch.from_numpy(y[domain][batch * batch_size: (batch + 1) * batch_size]).cuda())
            else:
                labels = Variable(torch.from_numpy(y[domain][batch * batch_size: (batch + 1) * batch_size]))

    if 'x' in kwargs.keys() and 'y' in kwargs.keys():
        return inputs, labels
    elif 'x' in kwargs.keys():
        return inputs
    elif 'y' in kwargs.keys():
        return labels

def train_maml():
    ########### config ############
    lr = 0.01
    meta_lr = 0.01
    epoch_num = 5000
    batch_size = 32
    
    model = Network()
    if torch.cuda.is_available():
        model.cuda()

    creiterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)

    ########### train ############
    for epoch in range(epoch_num):
        for batch in range(int(len(train_x) / batch_size) + 1):
            inputs = numpy_to_var(batch, batch_size, x=train_x, \
                                  last_batch=(batch == int(len(train_x) / batch_size)))

            init_state = copy.deepcopy(model.state_dict())
            loss_tasks = []

            for dom in range(d_sour_num):
                labels = numpy_to_var(batch, batch_size, domain=dom, y=train_y, \
                                      last_batch=(batch == int(len(train_x) / batch_size)))

                # # tmp-updated model for each domain
                model.load_state_dict(init_state)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = creiterion(outputs, labels)
                loss.backward()
                model.backward()

                # # compute the loss of tmp-updated model
                outputs = model(inputs)
                loss = creiterion(outputs, labels)

                # # record loss for diff domains
                loss_tasks.append(loss)

            model.load_state_dict(init_state)
            meta_optimizer.zero_grad()
            meta_loss = torch.stack(loss_tasks).sum(0) / d_sour_num
            meta_loss.backward()
            model.backward()

        ######################### validation ############################
        if epoch % 100 == 0:
            val_init_state = copy.deepcopy(model.state_dict())
            val_loss_tasks = []
            for dom in range(d_sour_num):
                val_inputs, val_labels = numpy_to_var(0, batch_size, x=val_x, y=val_y, domain=dom)

                # # tmp-updated model for each domain
                model.load_state_dict(val_init_state)
                optimizer.zero_grad()
                val_outputs = model(val_inputs)
                val_loss = creiterion(val_outputs, val_labels)
                val_loss.backward()
                model.backward()

                # # compute the loss of tmp-updated model
                val_outputs = model(val_inputs)
                val_loss = creiterion(val_outputs, val_labels)

                # # record loss for diff domains
                val_loss_tasks.append(val_loss)

                # inputs, labels = numpy_to_var(0, batch_size, x=val_x, y=val_y, domain=dom)
                # predicted = model(inputs)

            val_losses = torch.stack(val_loss_tasks).sum(0) / d_sour_num
            model.load_state_dict(val_init_state)

            print('epoch {}, validation loss {}'.format(epoch, val_losses.item()))

    torch.save(model.state_dict(), './model/model.pkl')

    ############ adaptation ######## 

    model.load_state_dict(torch.load('./model/model.pkl')) 
    for epoch in range(10):
        inputs, labels = numpy_to_var(0, batch_size, x=support_x, y=support_y)
        # if torch.cuda.is_available():
        #     inputs = Variable(torch.from_numpy(support_x).cuda())
        #     labels = Variable(torch.from_numpy(support_y[0]).cuda())
        # else:
        #     inputs = Variable(torch.from_numpy(support_x))
        #     labels = Variable(torch.from_numpy(support_y[0]))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = creiterion(outputs, labels)
        loss.backward()
        model.backward()

        # optimizer.step()

        if epoch % 1 == 0:
            print('epoch {}, adaptation loss {}'.format(epoch, loss.item()))



    ############ test #############    
    with torch.no_grad(): 
    # we don't need gradients in the testing phase
    
        if torch.cuda.is_available():
            outputs = model(Variable(torch.from_numpy(test_x).cuda()))
            labels = Variable(torch.from_numpy(test_y[0]).cuda())
        else:
            outputs = model(Variable(torch.from_numpy(test_x)))
            labels = Variable(torch.from_numpy(test_y[0]))
    test_loss = creiterion(outputs, labels)
    print('epoch {}, test loss {}'.format(epoch, test_loss.item()))

    plt.plot(test_x, test_y[0], 'bo')
    plt.plot(test_x, outputs.data.numpy(), 'ro')
    plt.savefig('regression_output.png')





def train():
    ########### config ############
    learning_rate = 0.01
    epoch_num = 10000
    batch_size = 32
    
    model = Network(learning_rate=learning_rate)
    if torch.cuda.is_available():
        model.cuda()

    creiterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    ########### train ############
    for epoch in range(epoch_num):
        for batch in range(int(len(train_x) / batch_size) + 1):
            inputs, labels = numpy_to_var(batch, batch_size, x=train_x, y=train_y,\
                                          last_batch=(batch == int(len(train_x) / batch_size)))


            optimizer.zero_grad()
            outputs = model(inputs)
            loss = creiterion(outputs, labels)
            loss.backward()
            model.backward()


        if epoch % 100 == 0:
            ######################### validation ############################
            with torch.no_grad():
                inputs, labels = numpy_to_var(0, batch_size, x=val_x, y=val_y)
                predicted = model(inputs)

                val_loss = creiterion(predicted, labels)

            print('epoch {}, loss {}'.format(epoch, val_loss.item()))


    ############ test ########    
    with torch.no_grad(): # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            predicted = model(Variable(torch.from_numpy(test_x_old).cuda())).cpu().data.numpy()
        else:
            predicted = model(Variable(torch.from_numpy(test_x_old))).data.numpy()

    # with open('./regression_output.pkl', 'wb') as f:
    #     pickle.dump([test_x, predicted, test_y], f)
    # pdb.set_trace()

    plt.plot(test_x_old, test_y_old[0], 'bo')
    plt.plot(test_x_old, predicted, 'ro')
    plt.savefig('regression_output.png')


def main():
    # train()
    train_maml()

if __name__ == "__main__":
    main()

