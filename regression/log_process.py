#!/usr/bin/env python3
#
import sys, os
import pdb
import matplotlib.pyplot as plt


target = 'maml'
train_log = './log/' + target + '_detail.log'

with open(train_log) as log_train:
    epoch     = []
    loss_meta = []
    loss_val  = []
    for line in log_train.readlines():
        words = line.replace(',','').split()
        epoch.append(int(words[1]))
        loss_meta.append(float(words[4])/10)
        loss_val.append(float(words[7]))

# plt.figure()
# plt.plot(epoch, loss_meta, 'r', label = 'train loss')
# plt.plot(epoch, loss_val, 'bo', label = 'validation loss')
# plt.legend(loc='upper center')
# plt.savefig('./result_pic/compare/' + target + '_train_log.png')
# plt.close()

test_log = './log/' + target + '_test.log'

with open(test_log) as log_test:
    adapt      = []
    loss_adapt = []
    loss_test  = []
    loss_adapt_epoch = []
    loss_test_epoch  = []
    for line in log_test.readlines():
        words = line.replace(',','').split()
        if words[1] == '0' and loss_adapt_epoch != []:
            loss_adapt.append(loss_adapt_epoch)
            loss_test.append(loss_test_epoch)
            loss_adapt_epoch = []
            loss_test_epoch =[]

        loss_adapt_epoch.append(float(words[3]))
        loss_test_epoch.append(float(words[5]))

    loss_adapt.append(loss_adapt_epoch)
    loss_test.append(loss_test_epoch)

    # pdb.set_trace()


shot_num = 30

# plt.figure()
# plt.plot(epoch, [i[shot_num] for i in loss_adapt], 'r', label = 'adapt loss')
# plt.plot(epoch, [i[shot_num] for i in loss_test], 'b', label = 'test loss')
# plt.title(str(shot_num) + '-shot')
# plt.legend(loc='upper center')
# plt.savefig('./result_pic/compare/' + target + '_test_log_' + str(shot_num) + 'shot.png')
# plt.close()


min_idx = loss_val.index(min(loss_val))
min_idx = 9
for min_idx in range(50):
    print(min_idx)

    plt.figure()
    plt.plot(loss_adapt[min_idx], 'r', label = 'adapt loss')
    plt.plot(loss_test[min_idx], 'b', label = 'test loss')
    plt.title('test on best model')
    plt.legend(loc='upper center')
    plt.savefig('./result_pic/compare/' + target + '_test_log_' +str(min_idx) + '.png')
    plt.close()
