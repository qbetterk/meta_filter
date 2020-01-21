#!/usr/bin/env python3
#
import os, sys
import math, random, copy, argparse, logging, json, time, csv
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pdb
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms

from config import global_config as cfg


class conv4(nn.Module):
    def __init__(self):
        super(conv4, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.linear = nn.Linear(64 * 5 * 5, 5)

    def forward(self, input_image):
        x = F.max_pool2d(F.batch_norm(F.relu(self.conv1(input_image))), 2, 2, 0)
        x = F.max_pool2d(F.batch_norm(F.relu(self.conv2(x))), 2, 2, 0)
        x = F.max_pool2d(F.batch_norm(F.relu(self.conv3(x))), 2, 2, 0)
        x = F.max_pool2d(F.batch_norm(F.relu(self.conv4(x))), 2, 1, 0)
        x = x.view(x.size(0), -1)
        return x

class read_data():
    def __init__(self, mode='test', batchsz=100, resize=84, startidx=0):
        self.train = {}
        self.val = {}
        self.test = {}
        self.csv_dir = "./data/"
        self.mode = mode

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = cfg.n_way  # n-way
        self.k_shot = cfg.support_num  # k-shot
        self.k_query = cfg.test_num  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        # print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        # mode, batchsz, n_way, k_shot, k_query, resize))


        self.image_dir = os.path.join(self.csv_dir, 'images')  # image path
        csvdata = self.loadCSV(os.path.join(self.csv_dir, mode + '.csv'))  # csv path
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

        flatten_support_x = [os.path.join(self.image_dir, item)
                             for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array(
            [self.img2label[item[:9]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [os.path.join(self.image_dir, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[item[:9]]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        # print('global:', support_y, query_y)
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



class Unknown():
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

        self.n_way = 5
        self.support_num = cfg.support_num
        self.model = conv4()
        if torch.cuda.is_available():
            self.model.cuda()

        self.creiterion = nn.MSELoss()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum = 0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr)

        self.resize = 84

        self.data = read_data()

def main():
    model = Unknown()










if __name__ == "__main__":
    main()