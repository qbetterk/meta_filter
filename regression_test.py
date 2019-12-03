#!/usr/bin/env python3
#
import os, sys
import math, random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import matplotlib.pyplot as plt
import numpy as np

random.seed(0)
"""
amplitude in [0.1, 5.0]
phase     in [0, pi]
x         in [-5.0, 5.0]
sample point 10
"""
random.seed(0)
torch.manual_seed(1)

d_sour_num = 20      # number of source domains
# d_sour_a = [np.random.uniform(1.0, 1.0) for _ in range(d_sour_num)]
d_sour_a = [random.uniform(0.1, 5.0) for _ in range(d_sour_num)]
d_sour_b = [np.random.uniform(0, np.pi) for _ in range(d_sour_num)]
d_targ_num = 1       # number of target domain
# d_targ_a = [np.random.uniform(1.0, 1.0) for _ in range(d_sour_num)]
d_targ_a = [random.uniform(0.1, 5.0) for _ in range(d_sour_num)]
d_targ_b = [np.random.uniform(0, np.pi) for _ in range(d_targ_num)]


train_num = 100     # number of training point in each domain
train_x   = np.array([np.random.uniform(-5.0, 5.0) for _ in range(train_num)], dtype=np.float32).reshape(-1,1)
train_y   = np.array([[d_sour_a[j] * np.sin(i + d_sour_b[j]) for i in train_x] for j in range(d_sour_num)], dtype=np.float32).reshape(d_sour_num, train_num, 1)

val_num = 100
val_x   = np.array([np.random.uniform(-5.0, 5.0) for _ in range(val_num)], dtype=np.float32).reshape(-1,1)
val_y   = np.array([[d_sour_a[j] * np.sin(i + d_sour_b[j]) for i in val_x] for j in range(d_sour_num)], dtype=np.float32).reshape(d_sour_num, val_num, 1)

support_num = 10
support_x   = np.array([np.random.uniform(-5.0, 5.0) for _ in range(support_num)], dtype=np.float32).reshape(-1,1)
support_y   = np.array([[d_targ_a[j] * np.sin(i + d_targ_b[j]) for i in support_x] for j in range(d_targ_num)], dtype=np.float32).reshape(d_targ_num, support_num, 1)

test_num = 100
test_x   = np.array([np.random.uniform(-5.0, 5.0) for _ in range(test_num)], dtype=np.float32).reshape(-1,1)
test_y   = np.array([[d_targ_a[j] * np.sin(i + d_targ_b[j]) for i in test_x] for j in range(d_targ_num)], dtype=np.float32).reshape(d_targ_num, test_num, 1)

test_x_old   = np.array([np.random.uniform(-5.0, 5.0) for _ in range(test_num)], dtype=np.float32).reshape(-1,1)
test_y_old   = np.array([[d_sour_a[j] * np.sin(i + d_sour_b[j]) for i in test_x_old] for j in range(d_sour_num)], dtype=np.float32).reshape(d_sour_num, test_num, 1)


class regression(nn.Module):
    """
    2 hiddden layers of size 40
    with ReLU
    step size alpha = 0.01
    """
    def __init__(self, input_size=1, output_size=1):
        super(regression, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
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

def numpy_to_var(batch, batch_size, last_batch=True, domain=0, **kwargs):
    """
    transfer numpy array to pytorch variable
    considering the batch and cuda
    """
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


def train():
    ########### config ############
    lr = 0.01
    meta_lr = 0.01
    epoch_num = 1000
    batch_size = 20
    sample_num = 20

    model = regression()
    if torch.cuda.is_available():
        model.cuda()

    creiterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pre_val_loss = 0
    converge_step = 5

    ########### train ############
    for epoch in range(epoch_num):
        sample_idx = np.random.choice(train_num, sample_num)
        for batch in range(math.ceil(sample_num / batch_size)):
            inputs = numpy_to_var(batch, batch_size, x = train_x[sample_idx], \
                                    last_batch = (batch == int(sample_num / batch_size)))

            labels = numpy_to_var(batch, batch_size, y = train_y[:, sample_idx], \
                                    last_batch = (batch == int(sample_num / batch_size)))

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = creiterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # pdb.set_trace()
            
            if math.isnan(loss.item()):
                pdb.set_trace()


        ######################### validation ############################
        if epoch % 100 == 0:
            val_loss = []
            val_inputs, val_labels = numpy_to_var(0, batch_size, x = val_x, y = val_y)

            optimizer.zero_grad()
            val_outputs = model(val_inputs)
            val_loss = creiterion(val_outputs, val_labels)
            print('epoch {}, validation loss {:f}'.format(epoch, val_loss.item()))

            if abs(pre_val_loss - val_loss.item()) < 1e-5:
                converge_step -= 1
            else:
                converge_step = 5

                pre_val_loss = val_loss.item()

            if converge_step == 0 or val_loss.item() < 1e-4:
                break

    torch.save(model.state_dict(), './model/model_filter.pkl')



    ############ test #############   
    model.load_state_dict(torch.load('./model/model_filter.pkl'))  
    with torch.no_grad(): 
    # we don't need gradients in the testing phase

        if torch.cuda.is_available():
            outputs = model(Variable(torch.from_numpy(test_x).cuda()))
            labels = Variable(torch.from_numpy(test_y[0]).cuda())
        else:
            outputs = model(Variable(torch.from_numpy(test_x)))
            labels = Variable(torch.from_numpy(test_y[0]))
        test_loss = creiterion(outputs, labels)
        # print('epoch {}, test loss {}'.format(epoch, test_loss.item()))


        print('epoch {}, test_loss {}'.format(epoch, test_loss.item()))



    plt.figure()
    plt.plot(test_x, test_y[0], 'bo')
    plt.plot(test_x, outputs.data.cpu().numpy(), 'ro')
    # plt.plot(support_x, support_y[0], 'go')
    plt.savefig('regression_output_adapt' + str(epoch+1) + 
                                    '_sn' + str(sample_num) + 
                                    '_bz' + str(batch_size) + 
                                 '_epoch' + str(epoch_num) + '.png')
    plt.close()


    # for epoch in range(epoch_num):
    #     sample_idx = np.random.choice(train_num, sample_num)
    #     for batch in range(math.ceil(sample_num / batch_size)):
    #         inputs = numpy_to_var(batch, batch_size, x=train_x[sample_idx], \
    #                               last_batch=(batch == int(sample_num / batch_size)))

    #         init_state = copy.deepcopy(model.state_dict())
    #         loss_tasks = []

    #         for dom in range(d_sour_num):
    #             labels = numpy_to_var(batch, batch_size, domain=dom, y=train_y[:, sample_idx], \
    #                                   last_batch=(batch == int(sample_num / batch_size)))

    #             # # tmp-updated model for each domain
    #             pdb.set_trace()
    #             model.load_state_dict(init_state)
    #             optimizer.zero_grad()
    #             outputs = model(inputs)
    #             loss = creiterion(outputs, labels)
    #             loss.backward()
    #             model.backward()

    #             # # compute the loss of tmp-updated model
    #             outputs = model(inputs)
    #             loss = creiterion(outputs, labels)

    #             # # record loss for diff domains
    #             loss_tasks.append(loss)

    #         model.load_state_dict(init_state)
    #         meta_optimizer.zero_grad()
    #         meta_loss = torch.stack(loss_tasks).sum(0) / d_sour_num
    #         meta_loss.backward()
    #         model.backward()

    #         if math.isnan(meta_loss.item()):
    #             pdb.set_trace()

    #     ######################### validation ############################
    #     if epoch % 100 == 0:
    #         val_init_state = copy.deepcopy(model.state_dict())
    #         val_loss_tasks = []
    #         for dom in range(d_sour_num):
    #             val_inputs, val_labels = numpy_to_var(0, batch_size, x=val_x, y=val_y, domain=dom)

    #             # # tmp-updated model for each domain
    #             model.load_state_dict(val_init_state)
    #             optimizer.zero_grad()
    #             val_outputs = model(val_inputs)
    #             val_loss = creiterion(val_outputs, val_labels)
    #             val_loss.backward()
    #             model.backward()

    #             # # compute the loss of tmp-updated model
    #             val_outputs = model(val_inputs)
    #             val_loss = creiterion(val_outputs, val_labels)

    #             # # record loss for diff domains
    #             val_loss_tasks.append(val_loss)

    #             # inputs, labels = numpy_to_var(0, batch_size, x=val_x, y=val_y, domain=dom)
    #             # predicted = model(inputs)

    #         val_losses = torch.stack(val_loss_tasks).sum(0) / d_sour_num
    #         model.load_state_dict(val_init_state)

    #         print('epoch {}, meta loss {:f}, validation loss {:f}'.format(epoch, meta_loss.item(), val_losses.item()))
    #         print(type(meta_loss.item()))

    #         if abs(pre_val_loss - val_losses.item()) < 1e-5:
    #             converge_step -= 1
    #         else:
    #             converge_step = 5

    #             pre_val_loss = val_losses.item()

    #         if converge_step == 0 or val_losses.item() < 1e-4:
    #             break

    # torch.save(model.state_dict(), './model/model_filter.pkl')

    # ############ adaptation ######## 

    # model.load_state_dict(torch.load('./model/model_filter.pkl')) 
    # for epoch in range(10):
    #     inputs, labels = numpy_to_var(0, batch_size, x=support_x, y=support_y)

    #     optimizer.zero_grad()
    #     outputs = model(inputs)
    #     adapt_loss = creiterion(outputs, labels)
    #     adapt_loss.backward()
    #     model.backward()

    #     # optimizer.step()

    #     # if epoch % 1 == 0:
    #     # print('epoch {}, adaptation loss {}'.format(epoch, adapt_loss.item()))



    #     ############ test #############    
    #     with torch.no_grad(): 
    #     # we don't need gradients in the testing phase

    #         if torch.cuda.is_available():
    #             outputs = model(Variable(torch.from_numpy(test_x).cuda()))
    #             labels = Variable(torch.from_numpy(test_y[0]).cuda())
    #         else:
    #             outputs = model(Variable(torch.from_numpy(test_x)))
    #             labels = Variable(torch.from_numpy(test_y[0]))
    #         test_loss = creiterion(outputs, labels)
    #         # print('epoch {}, test loss {}'.format(epoch, test_loss.item()))


    #         print('epoch {}, adapt_loss {}, test_loss {}'.format(epoch, adapt_loss.item(), test_loss.item()))










def main():
    train()


if __name__ == "__main__":
    main()