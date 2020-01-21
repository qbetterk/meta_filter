#!/usr/bin/env python3
#
import os, sys
import math, random, copy, argparse, logging, json, time
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from config import global_config as cfg


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
        self.lr = cfg.lr
        self.meta_lr = cfg.meta_lr
        self.filter_lr = cfg.filter_lr
        self.epoch_num = cfg.epoch_num
        self.val_period = cfg.val_period
        self.batch_size = cfg.batch_size
        self.sample_num = cfg.sample_num
        self.sample_num_val = cfg.sample_num_val
        self.domain_num = cfg.domain_num

        self.model = reg_net()
        if torch.cuda.is_available():
            self.model.cuda()

        self.creiterion = nn.MSELoss()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum = 0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr)

        self.pre_val_loss = cfg.pre_val_loss
        self.ear_stop_num = cfg.ear_stop_num
        self.ear_stop_num_test = cfg.ear_stop_num_test
        self.min_val_loss = cfg.min_val_loss

        self.model_dir = cfg.model_dir
        self.model_path = cfg.model_path
        self.filter_path = cfg.filter_path

        self.max_adapt_num = cfg.max_adapt_num

        self.generate_data()

    def generate_data(self):

        """
        amplitude in [0.1, 5.0]
        phase     in [0, pi]
        x         in [-5.0, 5.0]
        sample point 10
        """

        # cfg.d_sour_num = 20      # number of source domains
        self.d_sour_a = [np.random.uniform(0.1, 5.0) for _ in range(cfg.d_sour_num)]
        self.d_sour_b = [np.random.uniform(0, np.pi) for _ in range(cfg.d_sour_num)]
        # cfg.d_targ_num = 1       # number of target domain
        self.d_targ_a = [np.random.uniform(0.1, 5.0) for _ in range(cfg.d_targ_num)]
        self.d_targ_b = [np.random.uniform(0, np.pi) for _ in range(cfg.d_targ_num)]


        # cfg.train_num = 100     # number of training point in each domain
        self.train_x   = np.array([np.random.uniform(-5.0, 5.0) for _ in range(cfg.train_num)], dtype=np.float32).reshape(-1,1)
        self.train_y   = np.array([[self.d_sour_a[j] * np.sin(i + self.d_sour_b[j]) for i in self.train_x] for j in range(cfg.d_sour_num)], dtype=np.float32).reshape(cfg.d_sour_num, cfg.train_num, 1)

        # cfg.val_num = 100
        self.val_x   = np.array([np.random.uniform(-5.0, 5.0) for _ in range(cfg.val_num)], dtype=np.float32).reshape(-1,1)
        self.val_y   = np.array([[self.d_sour_a[j] * np.sin(i + self.d_sour_b[j]) for i in self.val_x] for j in range(cfg.d_sour_num)], dtype=np.float32).reshape(cfg.d_sour_num, cfg.val_num, 1)

        # cfg.support_num = 10
        self.support_x   = np.array([np.random.uniform(-5.0, 5.0) for _ in range(cfg.support_num)], dtype=np.float32).reshape(-1,1)
        self.support_y   = np.array([[self.d_targ_a[j] * np.sin(i + self.d_targ_b[j]) for i in self.support_x] for j in range(cfg.d_targ_num)], dtype=np.float32).reshape(cfg.d_targ_num, cfg.support_num, 1)

        # cfg.test_num = 100
        self.test_x   = np.array([np.random.uniform(-5.0, 5.0) for _ in range(cfg.test_num)], dtype=np.float32).reshape(-1,1)
        self.test_y   = np.array([[self.d_targ_a[j] * np.sin(i + self.d_targ_b[j]) for i in self.test_x] for j in range(cfg.d_targ_num)], dtype=np.float32).reshape(cfg.d_targ_num, cfg.test_num, 1)

        self.test_x_old  = np.array([np.random.uniform(-5.0, 5.0) for _ in range(cfg.test_num)], dtype=np.float32).reshape(-1,1)
        self.test_y_old  = np.array([[self.d_sour_a[j] * np.sin(i + self.d_sour_b[j]) for i in self.test_x_old] for j in range(cfg.d_sour_num)], dtype=np.float32).reshape(cfg.d_sour_num, cfg.test_num, 1)

    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        param_cnt = int(sum([np.prod(p.size()) for p in module_parameters]))

        print('total trainable params: %d' % param_cnt)
        return param_cnt

    def train(self):
        for epoch in range(self.epoch_num):
            sample_idx = np.random.choice(cfg.train_num, self.sample_num)
            for batch_idx in range(math.ceil(self.sample_num / self.batch_size)):
                inputs = numpy_to_var(batch_idx, self.batch_size, x = self.train_x[sample_idx], \
                                        last_batch = (batch_idx == int(self.sample_num / self.batch_size)))
                labels = numpy_to_var(batch_idx, self.batch_size, y = self.train_y[:, sample_idx], \
                                        last_batch = (batch_idx == int(self.sample_num / self.batch_size)))

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.creiterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            if epoch % 100 == 0:
                val_inputs, val_labels = numpy_to_var(0, self.batch_size, x = self.val_x, y = self.val_y)

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

    def test_(self, test_x_data = None, test_y_data = None, model_path = './model/basic.pkl'):
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
            sample_idx = np.random.choice(cfg.train_num, self.sample_num)
            for batch_idx in range(math.ceil(self.sample_num / self.batch_size)):
                inputs = numpy_to_var(batch_idx, self.batch_size, x = self.train_x[sample_idx], \
                                        last_batch = (batch_idx == int(self.sample_num / self.batch_size)))
                self.optimizer.zero_grad()
                loss_tasks = []
                for dom in range(cfg.d_sour_num):
                    labels = numpy_to_var(batch_idx, self.batch_size, domain = dom, y = self.train_y[:, sample_idx], \
                                            last_batch = (batch_idx == int(self.sample_num / self.batch_size)))

                    
                    outputs = self.model(inputs)
                    loss = self.creiterion(outputs, labels)
                    loss_tasks.append(loss)
                losses = torch.stack(loss_tasks).sum(0) / cfg.d_sour_num   
                losses.backward()
                self.optimizer.step()


            if epoch % 500 == 0:
                val_inputs, val_labels = numpy_to_var(0, self.batch_size, x = self.val_x, y = self.val_y)

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
        converge_step_left = self.ear_stop_num
        min_val_loss = self.min_val_loss
        sw = time.time()

        ################## train #####################
        for epoch in range(self.epoch_num):
            sample_idx = np.random.choice(cfg.train_num, self.sample_num)
            sample_idx_val = np.random.choice(cfg.train_num, self.sample_num_val)
            for batch_idx in range(math.ceil(self.sample_num / self.batch_size)):
                inputs = numpy_to_var(batch_idx, self.batch_size, x=self.train_x[sample_idx], \
                                      last_batch=(batch_idx == int(self.sample_num / self.batch_size)))
                labels_val = numpy_to_var(batch_idx, self.batch_size, x=self.train_x[sample_idx_val], \
                                      last_batch=(batch_idx == int(self.sample_num / self.batch_size)))

                init_state = copy.deepcopy(self.model.state_dict())
                loss_tasks = []

                domain_idx = random.choices(range(cfg.d_sour_num), k = self.domain_num)
                for dom in domain_idx:
                    labels = numpy_to_var(batch_idx, self.batch_size, domain=dom, y=self.train_y[:, sample_idx], \
                                          last_batch=(batch_idx == int(self.sample_num / self.batch_size)))
                    labels_val = numpy_to_var(batch_idx, self.batch_size, domain=dom, y=self.train_y[:, sample_idx_val], \
                                          last_batch=(batch_idx == int(self.sample_num / self.batch_size)))

                    # # tmp-updated model for each domain
                    self.model.load_state_dict(init_state)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.creiterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    # # compute the loss of tmp-updated model
                    outputs = self.model(labels_val)
                    loss = self.creiterion(outputs, labels_val)

                    # # record loss for diff domains with normalization
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
                for dom in domain_idx:
                    val_inputs, val_labels = numpy_to_var(0, self.batch_size, x=self.val_x, y=self.val_y, domain=dom)

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

                val_losses = torch.stack(val_loss_tasks).sum(0) / cfg.d_sour_num
                self.model.load_state_dict(val_init_state)

                logging.info('epoch {}, meta loss {:f}, validation loss {:f}, total time: {:.1f}min'.format(epoch, 
                                                            meta_loss.item(), val_losses.item(), (time.time()-sw)/60))

                # # save the model with the lowest validation loss
                if val_losses.item() < min_val_loss:
                    torch.save(self.model.state_dict(), self.model_path)
                    logging.info('mode saved')
                    min_val_loss = val_losses.item()
                    converge_step_left = self.ear_stop_num
                else:
                    converge_step_left -= 1
                    logging.info('early stop countdown %d' % converge_step_left)


                if converge_step_left == 0 or val_losses.item() < 1e-5:
                    return

    def test(self):
        '''
        return outputs
               test_loss_avg
        '''
        # outputs = []
        # test_losses = []
        # with torch.no_grad(): 
        # # we don't need gradients in the testing phase
        #     for i in range(cfg.d_targ_num):
        #         if torch.cuda.is_available():
        #             output = self.model(Variable(torch.from_numpy(self.test_x).cuda()))
        #             labels = Variable(torch.from_numpy(self.test_y[i]).cuda())
        #         else:
        #             output = self.model(Variable(torch.from_numpy(self.test_x)))
        #             labels = Variable(torch.from_numpy(self.test_y[i]))
        #         test_loss = self.creiterion(output, labels)
        #         outputs.append(output)
        #         test_losses.append(float(test_loss))
        #     # # average with normalization
        #     test_loss_avg = sum([test_losses[i] / self.d_targ_a[i] ** 2 for i in range(len(test_losses))]) / len(test_losses)
        # return outputs, test_loss_avg

        with torch.no_grad(): 
            if torch.cuda.is_available():
                output = self.model(Variable(torch.from_numpy(self.test_x).cuda()))
                labels = Variable(torch.from_numpy(self.test_y[0]).cuda())
            else:
                output = self.model(Variable(torch.from_numpy(self.test_x)))
                labels = Variable(torch.from_numpy(self.test_y[0]))
            test_loss = self.creiterion(output, labels)
        return output, test_loss

    def plot(self, outputs, file_path, mode = 'sin'):
        plt.figure()
        model = ''
        if 'maml' in file_path:
            model = 'maml'
        elif 'filter' in file_path:
            model = 'filter'

        title = mode + '_' + model +             \
                  '_sup'+ str(cfg.support_num) + \
                  '_sa' + str(self.sample_num) + \
                  '_bz' + str(self.batch_size) + \
                  '_do' + str(self.domain_num) + \
                  '_ep' + str(self.epoch_num)  + \
                  '_vp' + str(self.val_period)

        if mode == 'sin':
            plt.plot(self.test_x, self.test_y[0], 'bo', label = 'oracle')
            plt.plot(self.test_x, outputs.data.cpu().numpy(), 'ro', label = 'predicted')
            plt.plot(self.support_x, self.support_y[0], 'go', label = 'support point')

            title = file_path.split('.png')[0].split('_')[-1] + '_' + title
            plt.title(title)
        elif mode == 'loss':
            for losses in outputs:
                plt.plot(outputs[losses], label = losses)
            plt.xlabel('Adaptation Step Number')
            plt.ylabel('MSE')
            # plt.ylim(0, 0.5)
            plt.title(title)
        plt.legend()
        plt.savefig(file_path)
        plt.close()

    def test_maml(self):
        # # # check paths
        if not os.path.exists(self.model_path):
            raise ValueError("Model path does not exist")
        output_dir = os.path.join(self.model_dir, 'result_pic/')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_sub_dir = os.path.join(output_dir, 'support_num_' + str(cfg.support_num) + 
                                                  '_target_dom_' + str(cfg.d_targ_num))
        if not os.path.exists(output_sub_dir):
            os.mkdir(output_sub_dir)

        # # # load model
        self.model.load_state_dict(torch.load(self.model_path))
        
        # # # initilize parameters
        converge_step_left = self.ear_stop_num_test
        pre_adapt_loss = self.pre_val_loss
        min_val_loss = self.min_val_loss
        adapt_losses = []
        test_losses = []
        optimizer = self.optimizer

        for epoch in range(self.max_adapt_num + 1):
            file_name = 'adapt_step_' + str(epoch) + '.png'
            if epoch == 0:
                # # # zero-shot
                outputs, test_loss = self.test()
                print('epoch {}, adapt_loss {}, test_loss {}'.format(epoch, 0, test_loss.item()))
                self.plot(outputs, os.path.join(output_sub_dir, file_name))
            else:
                # # # adaptation 
                inputs, labels = numpy_to_var(0, self.batch_size, x=self.support_x, y=self.support_y)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                adapt_loss = self.creiterion(outputs, labels)
                adapt_loss.backward()
                optimizer.step()

                # # # test   
                outputs, test_loss = self.test()

                # # # record the losses for the later plotting
                adapt_losses.append(adapt_loss.item())
                test_losses.append(test_loss.item())

                # # # # print log and plot
                logging.info('epoch {}, adapt_loss {}, test_loss {}'.format(epoch, adapt_loss.item(), test_loss.item()))
                self.plot(outputs, os.path.join(output_sub_dir, file_name))

                if adapt_loss.item() < min_val_loss:
                    min_val_loss = adapt_loss.item()
                    converge_step_left = self.ear_stop_num_test
                else:
                    converge_step_left -= 1
                    logging.info('early stop countdown %d' % converge_step_left)

                # if converge_step_left == 0:
                #     break
                # if abs(adapt_loss.item() - pre_adapt_loss) / adapt_loss.item() < 1e-4:
                #     break
                # if abs(adapt_loss.item() - pre_adapt_loss) < 1e-4:
                #     break

                pre_adapt_loss = adapt_loss.item()

        self.plot({'adapt': adapt_losses, 'test': test_losses}, 
                   os.path.join(output_sub_dir, 'test_error.png'), mode='loss')

        pkl.dump({'adapt': adapt_losses, 'test': test_losses}, 
                 open(os.path.join(output_sub_dir, 'test_error.pkl'), 'wb'))

    def train_filter(self):
        converge_step_left = self.ear_stop_num
        min_val_loss = self.min_val_loss
        sw = time.time()

        #################### initialize the filter ###########################
        self.filter = reg_net()
        if torch.cuda.is_available():
            self.filter.cuda()
        filter_optimizer = optim.Adam(self.filter.parameters(), lr=self.filter_lr)

        ######################### train ###############################
        for epoch in range(self.epoch_num):
            optimizer = self.optimizer
            meta_optimizer = self.meta_optimizer
            sample_idx = np.random.choice(cfg.train_num, self.sample_num)
            sample_idx_val = np.random.choice(cfg.train_num, self.sample_num_val)

            for batch_idx in range(math.ceil(self.sample_num / self.batch_size)):
                inputs = numpy_to_var(batch_idx, self.batch_size, x = self.train_x[sample_idx], \
                                        last_batch = (batch_idx == int(self.sample_num / self.batch_size)))
                inputs_val = numpy_to_var(batch_idx, self.batch_size, x = self.train_x[sample_idx_val], \
                                        last_batch = (batch_idx == int(self.sample_num / self.batch_size)))

                init_state = copy.deepcopy(self.model.state_dict())
                domain_idx = random.choices(range(cfg.d_sour_num), k = self.domain_num)
                loss_tasks = []
                filter_grad = []

                for dom in domain_idx:
                    labels = numpy_to_var(batch_idx, self.batch_size, domain=dom, y=self.train_y[:, sample_idx], \
                                          last_batch=(batch_idx == int(self.sample_num / self.batch_size)))
                    labels_val = numpy_to_var(batch_idx, self.batch_size, domain=dom, y=self.train_y[:, sample_idx_val], \
                                          last_batch=(batch_idx == int(self.sample_num / self.batch_size)))

                    self.model.load_state_dict(init_state)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.creiterion(outputs, labels)
                    loss.backward(retain_graph=True)

                    params_gen = [p.grad.data for p in self.model.parameters()]
                    # # apply filter to gradient
                    for p, q in zip(self.model.parameters(), self.filter.parameters()):
                        p.grad.data = p.grad.data * q.data
                    optimizer.step()

                    optimizer.zero_grad()
                    outputs = self.model(inputs_val)
                    loss = self.creiterion(outputs, labels_val)
                    # # record with normalization
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

                filter_optimizer.zero_grad()
                for i, j in zip(self.filter.parameters(), filter_grad):
                    i.grad = copy.deepcopy(j)
                filter_optimizer.step()

                # # update meta 
                self.model.load_state_dict(init_state)
                meta_optimizer.zero_grad()
                meta_loss = torch.stack(loss_tasks).sum(0) / self.domain_num
                meta_loss.backward()
                meta_optimizer.step()

            ######################### validation ############################
            if epoch % self.val_period == 0:
                val_init_state = copy.deepcopy(self.model.state_dict())
                val_loss_tasks = []
                for dom in domain_idx:
                    val_inputs, val_labels = numpy_to_var(0, self.batch_size, x=self.val_x, y=self.val_y, domain=dom)

                    # # tmp-updated model for each domain
                    self.model.load_state_dict(val_init_state)
                    optimizer.zero_grad()
                    val_outputs = self.model(val_inputs)
                    val_loss = self.creiterion(val_outputs, val_labels)
                    val_loss.backward()
                    for p, q in zip(self.model.parameters(), self.filter.parameters()):
                        p.grad.data = p.grad.data * q.data
                    optimizer.step()

                    # # compute the loss of tmp-updated model
                    val_outputs = self.model(val_inputs)
                    val_loss = self.creiterion(val_outputs, val_labels)

                    # # record loss for diff domains
                    val_loss_tasks.append(val_loss)

                val_losses = torch.stack(val_loss_tasks).sum(0) / cfg.d_sour_num
                self.model.load_state_dict(val_init_state)

                logging.info('epoch {}, meta loss {:f}, validation loss {:f}, total time: {:.1f}min'.format(epoch, 
                                                            meta_loss.item(), val_losses.item(), (time.time()-sw)/60))

                # # save the model with the lowest validation loss
                if val_losses.item() < min_val_loss:
                    torch.save(self.model.state_dict(), self.model_path)
                    torch.save(self.filter.state_dict(), self.filter_path)
                    logging.info('mode saved')
                    min_val_loss = val_losses.item()
                    converge_step_left = self.ear_stop_num
                else:
                    converge_step_left -= 1
                    logging.info('early stop countdown %d' % converge_step_left)

                if converge_step_left == 0 or val_losses.item() < 1e-5:
                    return

    def test_filter(self):
        # # # check paths
        if not os.path.exists(self.model_path):
            print(self.model_path)
            pdb.set_trace()
            raise ValueError("Model path: " + self.model_path + "does not exist")
        output_dir = os.path.join(self.model_dir, 'result_pic/')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_sub_dir = os.path.join(output_dir, 'support_num_' + str(cfg.support_num) + 
                                                  '_target_dom_' + str(cfg.d_targ_num))
        if not os.path.exists(output_sub_dir):
            os.mkdir(output_sub_dir)

        # # # load model
        self.model.load_state_dict(torch.load(self.model_path))
        self.filter = reg_net()
        if torch.cuda.is_available():
            self.filter.cuda()
        self.filter.load_state_dict(torch.load(self.filter_path))
        
        # # # initilize parameters
        converge_step_left = self.ear_stop_num_test
        pre_adapt_loss = self.pre_val_loss
        min_val_loss = self.min_val_loss
        adapt_losses = []
        test_losses = []
        optimizer = self.optimizer

        for epoch in range(self.max_adapt_num + 1):
            file_name = 'adapt_step_' + str(epoch) + '.png'
            if epoch == 0:
                # # # zero-shot
                outputs, test_loss = self.test()
                logging.info('epoch {}, adapt_loss {}, test_loss {}'.format(epoch, 0, test_loss.item()))
                self.plot(outputs, os.path.join(output_sub_dir, file_name))
            else:
                # # # adaptation 
                inputs, labels = numpy_to_var(0, self.batch_size, x=self.support_x, y=self.support_y)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                adapt_loss = self.creiterion(outputs, labels)
                adapt_loss.backward()
                for p, q in zip(self.model.parameters(), self.filter.parameters()):
                    p.grad.data = p.grad.data * q.data
                optimizer.step()

                # # # test   
                outputs, test_loss = self.test()

                # # # record the losses for the later plotting
                adapt_losses.append(adapt_loss.item())
                test_losses.append(test_loss.item())

                # # # # print log and plot
                logging.info('epoch {}, adapt_loss {}, test_loss {}'.format(epoch, adapt_loss.item(), test_loss.item()))
                self.plot(outputs, os.path.join(output_sub_dir, file_name))

                if adapt_loss.item() < min_val_loss:
                    min_val_loss = adapt_loss.item()
                    converge_step_left = self.ear_stop_num_test
                else:
                    converge_step_left -= 1
                    logging.info('early stop countdown %d' % converge_step_left)

                if abs(adapt_loss.item() - pre_adapt_loss) / adapt_loss.item() < 1e-4:
                    converge_step_left -= 1
                    logging.info('early stop countdown %d' % converge_step_left)

                if abs(adapt_loss.item() - pre_adapt_loss) < 1e-4:
                    converge_step_left -= 1
                    logging.info('early stop countdown %d' % converge_step_left)

                # if converge_step_left <= 0:
                #     return

                pre_adapt_loss = adapt_loss.item()

        self.plot({'adapt': adapt_losses, 'test': test_losses}, 
                   os.path.join(output_sub_dir, 'test_error.png'), 
                   mode='loss')

        pkl.dump({'adapt': adapt_losses, 'test': test_losses}, 
                 open(os.path.join(output_sub_dir, 'test_error.pkl'), 'wb'))

def parse_arg_cfg(args):
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k=='cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train_filter')
    parser.add_argument('--cfg', nargs='*')
    args = parser.parse_args()

    if '_' in args.mode:
        cfg.mode = args.mode.split('_')[0]
        cfg.alg  = args.mode.split('_')[-1]
    else:
        cfg.mode = args.mode

    parse_arg_cfg(args)

    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    if cfg.model_dir == '':
        cfg.model_dir = 'experiments/{}_sd{}_lr{}_mlr{}_flr{}_sa{}_sav{}_dn{}_sd{}_es{}_est{}/'.format(cfg.alg,
                         cfg.seed, cfg.lr, cfg.meta_lr, cfg.filter_lr, cfg.sample_num, cfg.sample_num_val,
                         cfg.domain_num, cfg.d_sour_num, cfg.ear_stop_num, cfg.ear_stop_num_test)

    if cfg.mode == 'train':
        if not os.path.exists(cfg.model_dir):
            os.mkdir(cfg.model_dir)
        cfg.model_path = os.path.join(cfg.model_dir, 'model.pkl')
        cfg.filter_path = os.path.join(cfg.model_dir, 'model_filter.pkl')

    elif cfg.mode == 'test' or cfg.mode=='adjust':
        cfg_load = json.loads(open(os.path.join(cfg.model_dir, 'config.json'), 'r').read())
        for k, v in cfg_load.items():
            if k in dir(cfg):
                continue
            setattr(cfg, k, v)

    cfg._init_logging_handler(log_dir = cfg.model_dir)

    if cfg.cuda:
        torch.cuda.set_device(cfg.cuda_device)
        logging.info('Device: {}'.format(torch.cuda.current_device()))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    reg = Regression()

    cfg.model_parameters = reg.count_params()
    logging.info(str(cfg))

    if args.mode == 'train_maml':
        if cfg.save_log:
            with open(os.path.join(cfg.model_dir, 'config.json'), 'w') as f:
                json.dump(cfg.__dict__, f, indent=2)
        reg.train_maml()
        reg.test_maml()

    if args.mode == 'test_maml':
        reg.test_maml()

    if args.mode == 'train_filter':
        if cfg.save_log:
            with open(os.path.join(cfg.model_dir, 'config.json'), 'w') as f:
                json.dump(cfg.__dict__, f, indent=2)
        reg.train_filter()
        reg.test_filter()

    if args.mode == 'test_filter':
        reg.test_filter()

if __name__ == "__main__":
    main()