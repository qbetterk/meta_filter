#!/usr/bin/env python3
#
import sys, os
import matplotlib.pyplot as plt
import pickle as pkl
import argparse
from config import global_config as cfg



parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0)
parser.add_argument('--lr', default=cfg.lr)
parser.add_argument('--meta_lr', default=cfg.meta_lr)
parser.add_argument('--filter_lr', default=cfg.filter_lr)
parser.add_argument('--sample_num', default=cfg.sample_num)
parser.add_argument('--support_num', default=cfg.support_num)
parser.add_argument('--alg', type=str, default='')
parser.add_argument('--clip', default=1)
args = parser.parse_args()

# # # maml v.s. filter   for regression task # # # 

# # # format: [adapt_losses, test_losses]
# maml_pkl_path = './tmp/maml_epoch_15000_as140.pkl'
# filter_pkl_path = './tmp/filter_epoch_15000_as108.pkl'

seed = args.seed
lr = args.lr
meta_lr = args.meta_lr
filter_lr = args.filter_lr
sample_num = args.sample_num
support_num = args.support_num
clip = args.clip

maml_pkl_path = './experiments/maml{}_clip{}_sd{}_lr{}_mlr{}_flr{}_tus{}_sa{}_sav{}_dn{}_sd{}_es{}_est{}/result_pic/support_num_{}_target_dom_{}/test_error.pkl'.format(
                args.alg, clip, seed, lr, meta_lr, filter_lr, cfg.train_update_step, sample_num, cfg.sample_num_val,
                cfg.domain_num, cfg.d_sour_num, cfg.ear_stop_num, cfg.ear_stop_num_test,
                support_num, cfg.d_targ_num)
# filter_pkl_path = './experiments/filter_sd' + str(seed) + '_lr0.0001_mlr0.001_flr0.001_sa10_dn15_sd20_es5_est30/result_pic/support_num_3_target_dom_' + str(target_dom) + '/test_error.pkl'

if not os.path.exists(maml_pkl_path):

    maml_pkl_path = './experiments/maml7_clip1_sd{}_lr{}_mlr{}_flr0.01_sa{}_sav{}_dn{}_sd{}_es{}_est{}/result_pic/support_num_{}_target_dom_{}/test_error.pkl'.format(
                seed, lr, meta_lr, sample_num, cfg.sample_num_val,
                cfg.domain_num, cfg.d_sour_num, cfg.ear_stop_num, cfg.ear_stop_num_test,
                support_num, cfg.d_targ_num)



filter_pkl_path = './experiments/filter{}_clip{}_sd{}_lr{}_mlr{}_flr{}_tus{}_sa{}_sav{}_dn{}_sd{}_es{}_est{}/result_pic/support_num_{}_target_dom_{}/test_error.pkl'.format(
                args.alg, clip, seed, lr, meta_lr, filter_lr, cfg.train_update_step, sample_num, cfg.sample_num_val,
                cfg.domain_num, cfg.d_sour_num, cfg.ear_stop_num, cfg.ear_stop_num_test,
                support_num, cfg.d_targ_num)
fig_name   = './result_pic/compare' + str(args.alg) + '_' + \
             '_'.join(filter_pkl_path.split('/')[2].split('_')[1:]) + '_' + \
             filter_pkl_path.split('/')[4] + \
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