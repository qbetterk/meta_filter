#!/usr/bin/env python3
#
import sys, os
import matplotlib.pyplot as plt
import pickle as pkl


# # # maml v.s. filter   for regression task # # # 

# # # format: [adapt_losses, test_losses]
maml_pkl_path = './tmp/maml_epoch_15000_as140.pkl'
filter_pkl_path = './tmp/filter_epoch_15000_as108.pkl'

# maml_pkl_path = './tmp/maml_epoch_5000_as159.pkl'
# filter_pkl_path = './tmp/filter_epoch_5000_as111.pkl'

# maml_pkl_path = './tmp/maml_epoch_3000_as111.pkl'
# filter_pkl_path = './tmp/filter_epoch_3000_as111.pkl'
fig_name   = './result_pic/compare_' + '_'.join(maml_pkl_path.split('_')[1:3]) + '.png'

# maml_pkl_path = './tmp/maml_epoch_15000_as140.pkl'
# filter_pkl_path = './tmp/filter_epoch_5000_as111.pkl'
# fig_name   = './result_pic/compare_best.png'

maml_res   = pkl.load(open(maml_pkl_path, 'rb'))
filter_res = pkl.load(open(filter_pkl_path, 'rb'))
plt.figure()
plt.plot(maml_res['adapt'], 'r', label = 'maml_adapt')
plt.plot(maml_res['test'], 'r--', label = 'maml_test')
plt.plot(filter_res['adapt'], 'b', label = 'filter_adapt')
plt.plot(filter_res['test'], 'b--', label = 'filter_test')
plt.legend()
plt.savefig(fig_name)
plt.close()