#!/usr/bin/env python3
#
import logging, time, os, pdb

class _Config():
    def __init__(self):
        self.reg_init()

    def reg_init(self):

        self.seed = 0
        self.lr = 0.05
        self.meta_lr = 0.005
        self.filter_lr = 0.005

        self.hid_size = 40
        self.epoch_num = 50001
        self.val_period = 200
        self.batch_size = 32
        self.sample_num = 3
        self.sample_num_maml = 3
        self.sample_num_val = 100
        self.domain_num = 64

        self.pre_val_loss = 0
        self.min_val_loss = 1 << 30
        self.ear_stop_num = 10
        self.ear_stop_num_test = 30
        self.max_adapt_num = 500

        self.mode = 'train'
        self.model_dir = ''
        # self.model_path = ''
        # self.filter_path = ''

        self.cuda = True
        self.cuda_device = 0
        self.restart_all = True
        self.train_update_step = 1
        self.test_update_step = 500
        self.clip = 3
        self.optim = 'sgd'

        # # # params to generate data points
        self.d_sour_num = 100      # number of source domains
        self.d_targ_num = 1       # number of target domain
        self.train_num = 200     # number of training point in each domain
        self.test_num = 100
        self.val_num = 100
        self.support_num = self.sample_num

        # # # log file
        self.save_log = True
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())



    def _init_logging_handler(self, log_dir = './log'):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if self.save_log:
            log_path = os.path.join(log_dir, 'log_{}_{}.txt'.format(self.mode, self.log_time))
            file_handler = logging.FileHandler(log_path)
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        else:
            logging.basicConfig(handlers=[stderr_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()