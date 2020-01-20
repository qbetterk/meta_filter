#!/usr/bin/env python3
#
import sys, os
import matplotlib.pyplot as plt
import pickle as pkl
import argparse
from config import global_config as cfg



parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0)
args = parser.parse_args()

# # # maml v.s. filter   for regression task # # # 

# # # format: [adapt_losses, test_losses]
# maml_pkl_path = './tmp/maml_epoch_15000_as140.pkl'
# filter_pkl_path = './tmp/filter_epoch_15000_as108.pkl'

seed = args.seed

maml_pkl_path = './experiments/maml_sd{}_lr{}_mlr{}_flr{}_sa{}_sav{}_dn{}_sd{}_es{}_est{}/result_pic/support_num_{}_target_dom_{}/test_error.pkl'.format(
                seed, cfg.lr, cfg.meta_lr, cfg.filter_lr, cfg.sample_num, cfg.sample_num_val,
                cfg.domain_num, cfg.d_sour_num, cfg.ear_stop_num, cfg.ear_stop_num_test,
                cfg.support_num, cfg.d_targ_num)
# filter_pkl_path = './experiments/filter_sd' + str(seed) + '_lr0.0001_mlr0.001_flr0.001_sa10_dn15_sd20_es5_est30/result_pic/support_num_3_target_dom_' + str(target_dom) + '/test_error.pkl'

filter_pkl_path = './experiments/filter_sd{}_lr{}_mlr{}_flr{}_sa{}_sav{}_dn{}_sd{}_es{}_est{}/result_pic/support_num_{}_target_dom_{}/test_error.pkl'.format(
                seed, cfg.lr, cfg.meta_lr, cfg.filter_lr, cfg.sample_num, cfg.sample_num_val,
                cfg.domain_num, cfg.d_sour_num, cfg.ear_stop_num, cfg.ear_stop_num_test,
                cfg.support_num, cfg.d_targ_num)

fig_name   = './result_pic/compare_' + \
             '_'.join(maml_pkl_path.split('/')[2].split('_')[1:]) + '_' + \
             maml_pkl_path.split('/')[4] + \
             '.png'


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