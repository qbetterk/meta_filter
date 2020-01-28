import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
import pdb
from    learner import Learner
from    copy import deepcopy

from config import global_config as cfg


class Filter(nn.Module):
    """
    filter Learner
    """
    def __init__(self, config):
        """

        :param args:
        """
        super(Filter, self).__init__()

        self.update_lr = cfg.update_lr
        self.meta_lr = cfg.meta_lr
        self.filter_lr = 0.01
        self.n_way = cfg.n_way
        self.k_spt = cfg.k_spt
        self.k_qry = cfg.k_qry
        self.task_num = cfg.task_num
        self.update_step = cfg.update_step
        self.update_step_test = cfg.update_step_test


        self.net = Learner(config, cfg.imgc, cfg.imgsz)
        self.filter = deepcopy(self.net)

        self.optim = optim.Adam(self.net.parameters(), lr=self.update_lr)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.filter_optim = optim.Adam(self.filter.parameters(), lr=self.filter_lr)




    def clip_filter_(self, parameters, max_norm=1):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        # parameters = list(filter(lambda p: p is not None, parameters))
        max_norm = float(max_norm)
        for p in parameters:
            if p is not None:
                p = F.sigmoid(p)

        return


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        filter_grad = []

        for i in range(task_num):

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0] * p[2], zip(grad, self.net.parameters(), self.filter.parameters())))

            params_gen = [p.data for p in grad]

            # [setsz, nway]
            logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry[i])
            losses_q[1] += loss_q
            grad = torch.autograd.grad(loss_q, fast_weights, retain_graph=True)
            params_spe = [p.data for p in grad]

            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry[i]).sum().item()
            corrects[1] = corrects[1] + correct
            
            # # update filter
            if len(filter_grad) == 0:
                filter_grad = [- j * k for j, k in zip(params_gen, params_spe)]
            elif len(filter_grad) == len(params_gen):
                filter_grad = [i - j * k for i, j, k in zip(filter_grad, params_gen, params_spe)]
            else:
                raise ValueError("Invalid filter gradient: dimension mismatch")
                
            del params_gen, params_spe, grad, fast_weights, loss_q
            torch.cuda.empty_cache()

        self.filter_optim.zero_grad()
        for i, j in zip(self.filter.parameters(), filter_grad):
            i.grad = deepcopy(j)
        self.filter_optim.step()

        # self.clip_filter_(self.filter.parameters())

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()


        accs = np.array(corrects) / (querysz * task_num)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0] * p[2], zip(grad, net.parameters(), self.filter.parameters())))
        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0] * p[2], zip(grad, fast_weights, self.filter.parameters())))
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz

        return accs




def main():
    pass


if __name__ == '__main__':
    main()




            # # # # # # trial two
            # """
            # does not work
            # Test acc: [0.1998 0.1996 0.1996 0.1996 0.1996 0.1996 0.1996 0.1996 0.1996 0.1996]
            # """
            # self.optim.zero_grad()
            # logits = self.net(x_spt[i], self.net.parameters(), bn_training=True)
            # loss = F.cross_entropy(logits, y_spt[i])
            # loss.backward()

            # params_gen = [p.grad.data for p in self.net.parameters()]
            # # # apply filter to gradient
            # for p, q in zip(self.net.parameters(), self.filter.parameters()):
            #     p.grad.data = p.grad.data * q.data
            # self.optim.step()

            # self.optim.zero_grad()
            # logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
            # loss_q = F.cross_entropy(logits_q, y_qry[i])
            # losses_q[1] += loss_q

            # loss_q.backward(retain_graph=True)
            # params_spe = [p.grad.data for p in self.net.parameters()]


