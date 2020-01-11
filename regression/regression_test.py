#!/usr/bin/env python3
#
import os, sys
import math, random
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import pdb
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


"""
amplitude in [0.1, 5.0]
phase     in [0, pi]
x         in [-5.0, 5.0]
sample point 10
"""
random.seed(0)
np.random.seed(0)
torch.manual_seed(1)

d_sour_num = 20      # number of source domains
# d_sour_a = [np.random.uniform(1.0, 1.0) for _ in range(d_sour_num)]
d_sour_a = [np.random.uniform(0.1, 5.0) for _ in range(d_sour_num)]
d_sour_b = [np.random.uniform(0, np.pi) for _ in range(d_sour_num)]
d_targ_num = 1       # number of target domain
d_targ_a = [np.random.uniform(0.1, 5.0) for _ in range(d_targ_num)]
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


class filter_optim():

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                # state = self.state[p]

                # # State initialization
                # if len(state) == 0:
                #     state['step'] = 0
                #     # Exponential moving average of gradient values
                #     state['exp_avg'] = torch.zeros_like(p.data)
                #     # Exponential moving average of squared gradient values
                #     state['exp_avg_sq'] = torch.zeros_like(p.data)
                #     if amsgrad:
                #         # Maintains max of all exp. moving avg. of sq. grad. values
                #         state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                # exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                # if amsgrad:
                #     max_exp_avg_sq = state['max_exp_avg_sq']
                # beta1, beta2 = group['betas']

                # state['step'] += 1
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

                # if group['weight_decay'] != 0:
                #     grad.add_(group['weight_decay'], p.data)

                # # Decay the first and second moment running average coefficient
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # if amsgrad:
                #     # Maintains the maximum of all 2nd moment running avg. till now
                #     torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                #     # Use the max. for normalizing running avg. of gradient
                #     denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                # else:
                #     denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # step_size = group['lr'] / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

class reg_net(nn.Module):
    """
    2 hiddden layers of size 40
    with ReLU
    step size alpha = 0.01
    """
    def __init__(self, input_size=1, output_size=1):
        super(reg_net, self).__init__()
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

def numpy_to_var(batch_idx, batch_size, last_batch=True, domain=0, **kwargs):
    """
    transfer numpy array to pytorch variable
    considering the batch and cuda
    """
    if 'x' in kwargs.keys():
        x = kwargs['x']
        if last_batch:
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(x[batch_idx * batch_size:]).cuda())
            else:
                inputs = Variable(torch.from_numpy(x[batch_idx * batch_size:]))
        else:
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(x[batch_idx * batch_size: (batch_idx + 1) * batch_size]).cuda())
            else:
                inputs = Variable(torch.from_numpy(x[batch_idx * batch_size: (batch_idx + 1) * batch_size]))

    if 'y' in kwargs.keys():
        y = kwargs['y']
        if last_batch:
            if torch.cuda.is_available():
                labels = Variable(torch.from_numpy(y[domain][batch_idx * batch_size:]).cuda())
            else:
                labels = Variable(torch.from_numpy(y[domain][batch_idx * batch_size:]))
        else:
            if torch.cuda.is_available():
                labels = Variable(torch.from_numpy(y[domain][batch_idx * batch_size: (batch_idx + 1) * batch_size]).cuda())
            else:
                labels = Variable(torch.from_numpy(y[domain][batch_idx * batch_size: (batch_idx + 1) * batch_size]))

    if 'x' in kwargs.keys() and 'y' in kwargs.keys():
        return inputs, labels
    elif 'x' in kwargs.keys():
        return inputs
    elif 'y' in kwargs.keys():
        return labels

class Regression():
    def __init__(self):
        self.lr = 0.0001
        self.meta_lr = 0.001
        self.epoch_num = 50001
        self.val_period = 500
        self.batch_size = 20
        self.sample_num = support_num
        self.domain_num = 15

        self.model = reg_net()
        if torch.cuda.is_available():
            self.model.cuda()

        self.creiterion = nn.MSELoss()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum = 0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr)

        self.pre_val_loss = 0
        self.ear_stop_num = 5
        self.min_val_loss = 1<<30

        self.model_dir = ''
        # # self.model_dir = './model/maml/detail'

        # if not os.path.exists(self.model_dir):
        #     os.makedirs(self.model_dir)

        self.max_adapt_num = 500

    def train(self):
        for epoch in range(self.epoch_num):
            sample_idx = np.random.choice(train_num, self.sample_num)
            for batch_idx in range(math.ceil(self.sample_num / self.batch_size)):
                inputs = numpy_to_var(batch_idx, self.batch_size, x = train_x[sample_idx], \
                                        last_batch = (batch_idx == int(self.sample_num / self.batch_size)))
                labels = numpy_to_var(batch_idx, self.batch_size, y = train_y[:, sample_idx], \
                                        last_batch = (batch_idx == int(self.sample_num / self.batch_size)))

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.creiterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            if epoch % 100 == 0:
                val_inputs, val_labels = numpy_to_var(0, self.batch_size, x = val_x, y = val_y)

                self.optimizer.zero_grad()
                val_outputs = self.model(val_inputs)
                val_loss = self.creiterion(val_outputs, val_labels)

                if abs(self.pre_val_loss - val_loss.item()) < 1e-5:
                    self.converge_step -= 1
                else:
                    self.converge_step = 5

                    self.pre_val_loss = val_loss.item()

                if self.converge_step == 0 or val_loss.item() < 1e-4:
                    break

                print('epoch {}, validation loss {:f}, train loss {:f}'.format(epoch, val_loss.item(), loss.item()))

        torch.save(self.model.state_dict(), './model/basic.pkl')

    def test_(self, test_x_data = test_x, test_y_data = test_y, model_path = './model/basic.pkl'):
        self.model.load_state_dict(torch.load(model_path))  
        with torch.no_grad(): 
        # we don't need gradients in the testing phase

            if torch.cuda.is_available():
                outputs = self.model(Variable(torch.from_numpy(test_x_data).cuda()))
                labels = Variable(torch.from_numpy(test_y_data[0]).cuda())
            else:
                outputs = self.model(Variable(torch.from_numpy(test_x_data)))
                labels = Variable(torch.from_numpy(test_y_data[0]))
            test_loss = self.creiterion(outputs, labels)


            # print('epoch {}, test_loss {}'.format(epoch, test_loss.item()))

        plt.figure()
        plt.plot(test_x_data, test_y_data[0], 'bo')
        plt.plot(test_x_data, outputs.data.cpu().numpy(), 'ro')
        plt.savefig('./result_pic/basic' +
                                        '_sn' + str(self.sample_num) + 
                                        '_bz' + str(self.batch_size) + 
                                     '_epoch' + str(self.epoch_num) + '.png')
        plt.close()

    def train_transfer(self):
        for epoch in range(self.epoch_num):
            sample_idx = np.random.choice(train_num, self.sample_num)
            for batch_idx in range(math.ceil(self.sample_num / self.batch_size)):
                inputs = numpy_to_var(batch_idx, self.batch_size, x = train_x[sample_idx], \
                                        last_batch = (batch_idx == int(self.sample_num / self.batch_size)))
                self.optimizer.zero_grad()
                loss_tasks = []
                for dom in range(d_sour_num):
                    labels = numpy_to_var(batch_idx, self.batch_size, domain = dom, y = train_y[:, sample_idx], \
                                            last_batch = (batch_idx == int(self.sample_num / self.batch_size)))

                    
                    outputs = self.model(inputs)
                    loss = self.creiterion(outputs, labels)
                    loss_tasks.append(loss)
                losses = torch.stack(loss_tasks).sum(0) / d_sour_num   
                losses.backward()
                self.optimizer.step()


            if epoch % 500 == 0:
                val_inputs, val_labels = numpy_to_var(0, self.batch_size, x = val_x, y = val_y)

                self.optimizer.zero_grad()
                val_outputs = self.model(val_inputs)
                val_loss = self.creiterion(val_outputs, val_labels)

                if abs(self.pre_val_loss - val_loss.item()) < 1e-5:
                    self.converge_step -= 1
                else:
                    self.converge_step = 5

                    self.pre_val_loss = val_loss.item()

                if self.converge_step == 0 or val_loss.item() < 1e-4:
                    print("====should break====")
                    # break

                print('epoch {}, validation loss {:f}, train loss {:f}'.format(epoch, val_loss.item(), loss.item()))

        torch.save(self.model.state_dict(), './model/transfer.pkl')

    def train_maml(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        ################## train #####################
        for epoch in range(self.epoch_num):
            sample_idx = np.random.choice(train_num, self.sample_num)
            for batch_idx in range(math.ceil(self.sample_num / self.batch_size)):
                inputs = numpy_to_var(batch_idx, self.batch_size, x=train_x[sample_idx], \
                                      last_batch=(batch_idx == int(self.sample_num / self.batch_size)))

                init_state = copy.deepcopy(self.model.state_dict())
                loss_tasks = []

                domain_idx = random.choices(range(d_sour_num), k = self.domain_num)
                for dom in domain_idx:
                    labels = numpy_to_var(batch_idx, self.batch_size, domain=dom, y=train_y[:, sample_idx], \
                                          last_batch=(batch_idx == int(self.sample_num / self.batch_size)))

                    # # tmp-updated model for each domain
                    self.model.load_state_dict(init_state)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.creiterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    # # compute the loss of tmp-updated model
                    outputs = self.model(inputs)
                    loss = self.creiterion(outputs, labels)

                    # # record loss for diff domains
                    loss_tasks.append(loss)

                self.model.load_state_dict(init_state)
                self.meta_optimizer.zero_grad()
                meta_loss = torch.stack(loss_tasks).sum(0) / self.domain_num
                meta_loss.backward()
                self.meta_optimizer.step()

                if math.isnan(meta_loss.item()):
                    pdb.set_trace()

            ######################### validation ############################
            if epoch % self.val_period == 0:
                val_init_state = copy.deepcopy(self.model.state_dict())
                val_loss_tasks = []
                for dom in range(d_sour_num):
                    val_inputs, val_labels = numpy_to_var(0, self.batch_size, x=val_x, y=val_y, domain=dom)

                    # # tmp-updated model for each domain
                    self.model.load_state_dict(val_init_state)
                    self.optimizer.zero_grad()
                    val_outputs = self.model(val_inputs)
                    val_loss = self.creiterion(val_outputs, val_labels)
                    val_loss.backward()
                    self.optimizer.step()

                    # # compute the loss of tmp-updated model
                    val_outputs = self.model(val_inputs)
                    val_loss = self.creiterion(val_outputs, val_labels)

                    # # record loss for diff domains
                    val_loss_tasks.append(val_loss)

                    # inputs, labels = numpy_to_var(0, batch_size, x=val_x, y=val_y, domain=dom)
                    # predicted = model(inputs)

                val_losses = torch.stack(val_loss_tasks).sum(0) / d_sour_num
                self.model.load_state_dict(val_init_state)

                print('epoch {}, meta loss {:f}, validation loss {:f}'.format(epoch, meta_loss.item(), val_losses.item()))
                # torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'epoch_' + str(epoch) + '.pkl'))

                if abs(self.pre_val_loss - val_losses.item()) < 1e-5:
                    self.ear_stop_num -= 1
                else:
                    self.ear_stop_num = 5
                    self.pre_val_loss = val_losses.item()

                if self.ear_stop_num == 0 or val_losses.item() < 1e-4:
                    break

                if val_losses.item() < self.min_val_loss:
                    torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'best_epoch_' + str(epoch) + '.pkl'))
                    torch.save(self.filter.state_dict(), os.path.join(self.model_dir, 'best_epoch_' + str(epoch) + '_filter.pkl'))
                    self.min_val_loss = val_losses.item()
                    converge_step_left = self.ear_stop_num
                else:
                    converge_step_left -= 1

                # # judge whether to stop
                # if abs(self.pre_val_loss - val_losses.item()) < 1e-4 * val_losses.item():
                #     self.ear_stop_num -= 1
                # else:
                #     self.ear_stop_num = 5

                #     self.pre_val_loss = val_losses.item()

                if converge_step_left == 0 or val_losses.item() < 1e-4:
                    return

        torch.save(self.model.state_dict(), './model/maml_earlystop.pkl')

    def test(self):
        with torch.no_grad(): 
        # we don't need gradients in the testing phase
            if torch.cuda.is_available():
                outputs = self.model(Variable(torch.from_numpy(test_x).cuda()))
                labels = Variable(torch.from_numpy(test_y[0]).cuda())
            else:
                outputs = self.model(Variable(torch.from_numpy(test_x)))
                labels = Variable(torch.from_numpy(test_y[0]))
            test_loss = self.creiterion(outputs, labels)
        return outputs, test_loss

    def plot(self, outputs, file_path, mode = 'sin'):
        plt.figure()
        model = ''
        if 'maml' in file_path:
            model = 'maml'
        elif 'filter' in file_path:
            model = 'filter'

        title = mode + '_' + model +             \
                  '_val'+ str(val_num) +         \
                  '_sup'+ str(support_num) +     \
                  '_sa' + str(self.sample_num) + \
                  '_bz' + str(self.batch_size) + \
                  '_do' + str(self.domain_num) + \
                  '_ep' + str(self.epoch_num)  + \
                  '_vp' + str(self.val_period)

        if mode == 'sin':
            plt.plot(test_x, test_y[0], 'bo', label = 'oracle')
            plt.plot(test_x, outputs.data.cpu().numpy(), 'ro', label = 'predicted')
            plt.plot(support_x, support_y[0], 'go', label = 'support point')

            title = file_path.split('.png')[0].split('_')[-1] + '_' + title
            plt.title(title)
        elif mode == 'loss':
            for losses in outputs:
                plt.plot(outputs[losses], label = losses)
            plt.xlabel('adaptation step')
            plt.ylabel('loss')
            plt.ylim(0, 0.5)
            plt.title(title)
        plt.legend()
        plt.savefig(file_path)
        plt.close()

    def test_adapt(self, model_path = None, output_dir = './result_pic/'):
        # # # check paths
        if model_path == None:
            raise ValueError("Lack of model path")
        if not os.path.exists(model_path):
            raise ValueError("Model path does not exist")
        output_sub_dir = os.path.join(output_dir, '_val'+ str(val_num) +
                                                  '_sup'+ str(support_num) +
                                                  '_sa' + str(self.sample_num) + 
                                                  '_bz' + str(self.batch_size) + 
                                                  '_do' + str(self.domain_num) + 
                                                  '_ep' + str(self.epoch_num)  +
                                                  '_vp' + str(self.val_period))
        if not os.path.exists(output_sub_dir):
            os.mkdir(output_sub_dir)

        # # # initilize parameters
        self.ear_stop_num = 30
        converge_step_left = self.ear_stop_num
        pre_adapt_loss = 0
        adapt_losses = []
        test_losses = []

        # # # load model 
        self.model.load_state_dict(torch.load(model_path)) 
        
        for epoch in range(self.max_adapt_num + 1):

            file_name = model_path.split('/')[-1].split('.')[0] + '_as' + str(epoch) + '.png'
            if epoch == 0:
                # # # zero-shot
                outputs, test_loss = self.test()
                print('epoch {}, adapt_loss {}, test_loss {}'.format(epoch, 0, test_loss.item()))
                self.plot(outputs, os.path.join(output_sub_dir, file_name))
            else:
                # # # adaptation 
                inputs, labels = numpy_to_var(0, self.batch_size, x=support_x, y=support_y)

                self.meta_optimizer.zero_grad()
                outputs = self.model(inputs)
                adapt_loss = self.creiterion(outputs, labels)
                adapt_loss.backward()
                self.meta_optimizer.step()

                # # # test   
                outputs, test_loss = self.test()

                # # # record the losses for the later plotting
                adapt_losses.append(adapt_loss.item())
                test_losses.append(test_loss.item())

                if epoch == 1:
                    print('epoch {}, adapt_loss {}, test_loss {}'.format(epoch, adapt_loss.item(), test_loss.item()))

                if adapt_loss.item() < self.min_val_loss:
                    self.min_val_loss = adapt_loss.item()
                    converge_step_left = self.ear_stop_num
                    if epoch % 1 == 0:
                        print('epoch {}, adapt_loss {}, test_loss {}'.format(epoch, adapt_loss.item(), test_loss.item()))
                    self.plot(outputs, os.path.join(output_sub_dir, file_name))
                else:
                    converge_step_left -= 1
                    print('early stop countdown %d' % converge_step_left)

                # # # when to stop early
                if converge_step_left == 0:
                    break
                if abs(adapt_loss.item() - pre_adapt_loss) / adapt_loss.item() < 1e-3:
                    break

                pre_adapt_loss = adapt_loss.item()

        self.plot({'adapt': adapt_losses, 'test': test_losses}, 
                   os.path.join(output_sub_dir, 'test_loss_' + model_path.split('/')[-1].split('.')[0] + '.png'), 
                   mode='loss')

        pkl.dump({'adapt': adapt_losses, 'test': test_losses}, open('./tmp/' + 
                                                   self.model_dir.split('/')[-2] + '_' + 
                                                   file_name.split('.png')[0] + 
                                                   '.pkl', 'wb'))

    def train_filter(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        converge_step_left = self.ear_stop_num

        #################### initialize the filter ###########################
        self.filter = reg_net()
        if torch.cuda.is_available():
            self.filter.cuda()
        self.filter_lr = 0.01
        self.filter_optimizer = optim.Adam(self.filter.parameters(), lr=self.filter_lr)

        ######################### train ###############################
        for epoch in range(self.epoch_num):
            sample_idx = np.random.choice(train_num, self.sample_num)
            for batch_idx in range(math.ceil(self.sample_num / self.batch_size)):
                inputs = numpy_to_var(batch_idx, self.batch_size, x = train_x[sample_idx], \
                                        last_batch = (batch_idx == int(self.sample_num / self.batch_size)))
                init_state = copy.deepcopy(self.model.state_dict())
                domain_idx = random.choices(range(d_sour_num), k = self.domain_num)

                loss_tasks = []
                filter_grad = []
                for dom in domain_idx:
                    labels = numpy_to_var(batch_idx, self.batch_size, domain=dom, y=train_y[:, sample_idx], \
                                          last_batch=(batch_idx == int(self.sample_num / self.batch_size)))

                    self.model.load_state_dict(init_state)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.creiterion(outputs, labels)
                    loss.backward(retain_graph=True)

                    params_gen = [p.grad.data for p in self.model.parameters()]
                    # # apply filter to gradient
                    for p, q in zip(self.model.parameters(), self.filter.parameters()):
                        p.grad.data = p.grad.data * q.data
                    self.optimizer.step()

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.creiterion(outputs, labels)
                    loss_tasks.append(loss)
                    loss.backward(retain_graph=True)
                    params_spe = [p.grad.data for p in self.model.parameters()]

                    # # update filter
                    if len(filter_grad) == 0:
                        filter_grad = [- j * k for j, k in zip(params_gen, params_spe)]
                    elif len(filter_grad) == len(params_gen):
                        filter_grad = [i - j * k for i, j, k in zip(filter_grad, params_gen, params_spe)]
                    else:
                        raise ValueError("Invalid filter gradient: dimension mismatch")

                self.filter_optimizer.zero_grad()
                for i, j in zip(self.filter.parameters(), filter_grad):
                    i.grad = copy.deepcopy(j)
                self.filter_optimizer.step()

                # # update meta 
                self.model.load_state_dict(init_state)
                self.meta_optimizer.zero_grad()
                meta_loss = torch.stack(loss_tasks).sum(0) / self.domain_num
                meta_loss.backward()
                self.meta_optimizer.step()

            ######################### validation ############################
            if epoch % self.val_period == 0:
                val_init_state = copy.deepcopy(self.model.state_dict())
                val_loss_tasks = []
                for dom in range(d_sour_num):
                    val_inputs, val_labels = numpy_to_var(0, self.batch_size, x=val_x, y=val_y, domain=dom)

                    # # tmp-updated model for each domain
                    self.model.load_state_dict(val_init_state)
                    self.optimizer.zero_grad()
                    val_outputs = self.model(val_inputs)
                    val_loss = self.creiterion(val_outputs, val_labels)
                    val_loss.backward()
                    for p, q in zip(self.model.parameters(), self.filter.parameters()):
                        p.grad.data = p.grad.data * q.data
                    self.optimizer.step()

                    # # compute the loss of tmp-updated model
                    val_outputs = self.model(val_inputs)
                    val_loss = self.creiterion(val_outputs, val_labels)

                    # # record loss for diff domains
                    val_loss_tasks.append(val_loss)

                val_losses = torch.stack(val_loss_tasks).sum(0) / d_sour_num
                self.model.load_state_dict(val_init_state)

                # # save the model with the lowest validation loss
                if val_losses.item() < self.min_val_loss:
                    torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'best_epoch_' + str(epoch) + '.pkl'))
                    torch.save(self.filter.state_dict(), os.path.join(self.model_dir, 'best_epoch_' + str(epoch) + '_filter.pkl'))
                    self.min_val_loss = val_losses.item()
                    converge_step_left = self.ear_stop_num
                else:
                    converge_step_left -= 1

                # # judge whether to stop
                # if abs(self.pre_val_loss - val_losses.item()) < 1e-4 * val_losses.item():
                #     self.ear_stop_num -= 1
                # else:
                #     self.ear_stop_num = 5

                #     self.pre_val_loss = val_losses.item()

                if converge_step_left == 0 or val_losses.item() < 1e-4:
                    return

                print('epoch {}, meta loss {:f}, validation loss {:f}'.format(epoch, meta_loss.item(), val_losses.item()))
                # torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'epoch_' + str(epoch) + '.pkl'))
                # torch.save(self.filter.state_dict(), os.path.join(self.model_dir, 'epoch_' + str(epoch) + '_filter.pkl'))

    def test_filter(self, model_path = None, output_dir = './result_pic/', save_outputs = False):
        # # # check paths
        if model_path == None:
            raise ValueError("Lack of model path")
        if not os.path.exists(model_path):
            raise ValueError("Model path does not exist")
        output_sub_dir = os.path.join(output_dir, '_val'+ str(val_num) +
                                                  '_sup'+ str(support_num) +
                                                  '_sa' + str(self.sample_num) + 
                                                  '_bz' + str(self.batch_size) + 
                                                  '_do' + str(self.domain_num) + 
                                                  '_ep' + str(self.epoch_num)  +
                                                  '_vp' + str(self.val_period))
        if not os.path.exists(output_sub_dir):
            os.mkdir(output_sub_dir)

        # # # load model
        self.model.load_state_dict(torch.load(model_path))
        self.filter = reg_net()
        if torch.cuda.is_available():
            self.filter.cuda()
        filter_path = model_path.split('.pkl')[0] + '_filter.pkl'
        self.filter.load_state_dict(torch.load(filter_path))
        
        # # # initilize parameters
        self.ear_stop_num = 30
        converge_step_left = self.ear_stop_num
        pre_adapt_loss = 0
        adapt_losses = []
        test_losses = []

        for epoch in range(self.max_adapt_num + 1):
            file_name = model_path.split('/')[-1].split('.')[0] + '_as' + str(epoch) + '.png'
            if epoch == 0:
                # # # zero-shot
                outputs, test_loss = self.test()
                print('epoch {}, adapt_loss {}, test_loss {}'.format(epoch, 0, test_loss.item()))
                self.plot(outputs, os.path.join(output_sub_dir, file_name))
            else:
                # # # adaptation 
                inputs, labels = numpy_to_var(0, self.batch_size, x=support_x, y=support_y)

                self.meta_optimizer.zero_grad()
                outputs = self.model(inputs)
                adapt_loss = self.creiterion(outputs, labels)
                adapt_loss.backward()
                for p, q in zip(self.model.parameters(), self.filter.parameters()):
                    p.grad.data = p.grad.data * q.data
                self.meta_optimizer.step()

                # # # test   
                outputs, test_loss = self.test()

                # # # record the losses for the later plotting
                adapt_losses.append(adapt_loss.item())
                test_losses.append(test_loss.item())

                # # # # print log and plot
                if epoch == 1: 
                    print('epoch {}, adapt_loss {}, test_loss {}'.format(epoch, adapt_loss.item(), test_loss.item()))

                if adapt_loss.item() < self.min_val_loss:
                    self.min_val_loss = adapt_loss.item()
                    converge_step_left = self.ear_stop_num
                    if epoch % 1 == 0: 
                        print('epoch {}, adapt_loss {}, test_loss {}'.format(epoch, adapt_loss.item(), test_loss.item()))
                    self.plot(outputs, os.path.join(output_sub_dir, file_name))
                else:
                    converge_step_left -= 1
                    print('early stop countdown %d' % converge_step_left)

                if converge_step_left == 0:
                    break
                if abs(adapt_loss.item() - pre_adapt_loss) / adapt_loss.item() < 1e-3:
                    break

                pre_adapt_loss = adapt_loss.item()

        self.plot({'adapt': adapt_losses, 'test': test_losses}, 
                   os.path.join(output_sub_dir, 'test_loss_' + model_path.split('/')[-1].split('.')[0] + '.png'), 
                   mode='loss')

        pkl.dump({'adapt': adapt_losses, 'test': test_losses}, open('./tmp/' + 
                                                   self.model_dir.split('/')[-2] + '_' + 
                                                   file_name.split('.png')[0] + 
                                                   '.pkl', 'wb'))


def main():
    reg = Regression()
    # reg.train()
    # reg.test(test_x_old, test_y_old)
    # reg.train_transfer()
    # reg.test_adapt(model_path = './model/transfer.pkl')
    # reg.test(model_path = './model/transfer.pkl')

    # reg.model_dir = './model/maml/N' + str(support_num) + '_detail'
    # # reg.model_dir = './model/maml/N3_detail'
    # # # reg.train_maml()
    # # for model_name in [f for f in os.listdir(reg.model_dir) if os.path.isfile(os.path.join(reg.model_dir, f))]:
    # #     reg.test_adapt(model_path = os.path.join(reg.model_dir, model_name), output_dir = './result_pic/maml/')
    # reg.test_adapt(model_path = os.path.join(reg.model_dir, 'epoch_42500.pkl'), output_dir = './result_pic/maml/')

    # reg.model_dir = './model/filter/N3_detail'
    reg.model_dir = './model/filter/N' + str(support_num) + '_detail'
    # reg.train_filter()
    # for model_name in [f for f in os.listdir(reg.model_dir) if os.path.isfile(os.path.join(reg.model_dir, f)) and not f.endswith('filter.pkl')]:
    #     reg.test_filter(model_path = os.path.join(reg.model_dir, model_name), output_dir = './result_pic/filter/')
    reg.test_filter(model_path = os.path.join(reg.model_dir, 'epoch_12500.pkl'), output_dir = './result_pic/filter/')


if __name__ == "__main__":
    main()