#!/usr/bin/env python3
#
import os, sys
import math, random, copy, argparse, logging, json, time, csv
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pdb
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from config import global_config as cfg

from MiniImagenet import MiniImagenet
from meta import Meta
from filter import Filter

class conv4(nn.Module):
    def __init__(self):
        super(conv4, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.linear = nn.Linear(64 * 5 * 5, 5)

    def forward(self, input_image):
        x = F.max_pool2d(self.conv1_bn(F.relu(self.conv1(input_image))), 2, 2, 0)
        x = F.max_pool2d(self.conv2_bn(F.relu(self.conv2(x))), 2, 2, 0)
        x = F.max_pool2d(self.conv3_bn(F.relu(self.conv3(x))), 2, 2, 0)
        x = F.max_pool2d(self.conv4_bn(F.relu(self.conv4(x))), 2, 1, 0)
        x = x.view(x.size(0), -1)
        return x

class read_data():
    def __init__(self, mode='test', batchsz=100, resize=84, startidx=0):
        self.csv_dir = "./data/"
        self.mode = mode

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = cfg.n_way  # n-way
        self.k_shot = cfg.k_spt  # k-shot
        self.k_query = cfg.k_qry  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx


        self.path = os.path.join(self.csv_dir, 'images')  # image path
        csvdata = self.loadCSV(os.path.join(self.csv_dir, self.mode + '.csv'))  # csv path
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
            self.img2label[k] = i + self.startidx  # {"img_name[:9]":label}
        self.cls_num = len(self.data)

        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                             transforms.Resize((self.resize, self.resize)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                             ])
        self.create_batch(self.batchsz)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int)

        flatten_support_x = [os.path.join(self.path, item)
                             for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array(
            [self.img2label[item[:9]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [os.path.join(self.path, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[item[:9]]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)
        # print(support_set_y)
        # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

    def lab2img(self, mode):
        """
        return: {label: [img1, img2, ...]}
        """
        lab2img = defaultdict(list)
        csv_path = os.path.join(self.csv_dir, mode + '.csv')
        raw_data = csv.reader(open(csv_path))
        for i, row in enumerate(raw_data):
            # # # row = [str(image.jpg), str(label)]
            if row[0].endswith('jpg'):
                lab2img[row[1]].append(row[0])
        return lab2img



class Model():
    def __init__(self):

        self.n_way = cfg.n_way
        # self.net = conv4()
        # if torch.cuda.is_available():
        #     self.net.cuda()
        self.device = torch.device('cuda:'+ str(cfg.cuda_device))

        self.data = read_data()

        self.config = [
            ('conv2d', [32, 3, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 1, 0]),
            ('flatten', []),
            ('linear', [cfg.n_way, 32 * 5 * 5])
        ]

        if cfg.alg == 'maml':
            self.net = Meta(self.config).to(self.device)
        elif cfg.alg == 'filter':
            self.net = Filter(self.config).to(self.device)

    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        param_cnt = int(sum([np.prod(p.size()) for p in module_parameters]))
        print('total trainable params: %d' % param_cnt)
        return param_cnt

    def train_maml(self):

        self.test_data = read_data(mode='test', batchsz=100)

        for epoch in range(100):
            # self.train_data = read_data(mode='train', batchsz=100)
            # # fetch meta_batchsz num of episode each time
            # print('epoch ' + str(epoch) + ' starting ... ')
            # db = DataLoader(self.train_data, self.n_way, shuffle=True, num_workers=1, pin_memory=True)

            # for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            #     x_spt, y_spt, x_qry, y_qry = x_spt.to(self.device),\
            #                                  y_spt.to(self.device),\
            #                                  x_qry.to(self.device),\
            #                                  y_qry.to(self.device)
            #     # pdb.set_trace()
            #     accs = self.net(x_spt, y_spt, x_qry, y_qry)

            #     if step % 20 == 0:
            #         print('step:', step, '\ttraining acc:', accs)
            # print('step:', step, '\ttraining acc:', accs)

            db_test = DataLoader(self.test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
            accs_all_test = []

            for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
                x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(self.device), \
                                             y_spt.squeeze(0).to(self.device), \
                                             x_qry.squeeze(0).to(self.device), \
                                             y_qry.squeeze(0).to(self.device)
                # if len(x_spt.shape) != 4:
                #     print(len(x_spt.shape))
                #     pdb.set_trace()
                accs = self.net.finetunning(x_spt, y_spt, x_qry, y_qry)
                accs_all_test.append(accs)

            # [b, update_step+1]
            accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            print('At step:', step, 'Test acc:', accs)


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
    parser.add_argument('--mode', default='train_maml')
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
        cfg.model_dir = 'experiments/{}_sd{}_lr{}_mlr{}_flr{}/'.format(cfg.alg,
                         cfg.seed, cfg.lr, cfg.meta_lr, cfg.filter_lr)

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

    model = Model()

    cfg.model_parameters = model.count_params()
    logging.info(str(cfg))

    model.train_maml()




if __name__ == "__main__":
    main()