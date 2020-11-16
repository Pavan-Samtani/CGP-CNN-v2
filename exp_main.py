#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import pandas as pd
import os

from cgp import *
from cgp_config import *
from cnn_train import CNN_train

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evolving CAE structures')
    parser.add_argument('--gpu_num', '-g', type=int, default=1, help='Num. of GPUs')
    parser.add_argument('--lam', '-l', type=int, default=2, help='Num. of offsprings')
    parser.add_argument('--log_path', default='./', help='Log path name')
    parser.add_argument('--mode', '-m', default='evolution', help='Mode (evolution / retrain / reevolution)')
    parser.add_argument('--init', '-i', action='store_true')
    parser.add_argument('--reduced', '-r', action='store_true', help="Whether to use reduced dataset version")
    parser.add_argument('--bias', '-b', default=0, type=float, 
                        help="Keep individual at least with (parent - bias) % accuracy if lesser macs")
    parser.add_argument('--epoch_load', '-e', type=int, default=0, 
                        help="In retrain mode, specifies with epoch to load for, default last")
                                                
    args = parser.parse_args()

    # --- Optimization of the CNN architecture ---
    if args.mode == 'evolution':
        # Create CGP configuration and save network information
        network_info = CgpInfoConvSet(rows=5, cols=30, level_back=10, min_active_num=1, max_active_num=30)
        with open(os.path.join(args.log_path, 'network_info.pkl'), mode='wb') as f:
            pickle.dump(network_info, f)
        # Evaluation function for CGP (training CNN and return validation accuracy)
        imgSize = 32
        eval_f = CNNEvaluation(gpu_num=args.gpu_num, dataset='cifar10', reduced=args.reduced, verbose=True, epoch_num=50, 
                               batchsize=128, imgSize=imgSize)

        # Execute evolution
        cgp = CGP(network_info, eval_f, lam=args.lam, imgSize=imgSize, init=args.init, bias=args.bias)
        cgp.modified_evolution(max_eval=250, mutation_rate=0.1, log_path=args.log_path)

    # --- Retraining evolved architecture ---
    elif args.mode == 'retrain':
        print('Retrain')
        # In the case of existing log_cgp.txt
        # Load CGP configuration
        with open(os.path.join(args.log_path, 'network_info.pkl'), mode='rb') as f:
            network_info = pickle.load(f)
        # Load network architecture
        cgp = CGP(network_info, None)
        data = pd.read_csv(os.path.join(args.log_path, 'log_cgp.txt'), header=None)  # Load log file
        cgp.load_log(list(data.iloc[[args.epoch_load - 1]].values.flatten().astype(int)))  # Read the log at final generation
        print(cgp._log_data(net_info_type='active_only', start_time=0))
        # Retraining the network
        temp = CNN_train('cifar10', reduced=args.reduced, validation=False, verbose=True, batchsize=128)
        acc, macs = temp(cgp.pop[0].active_net_list(), 0, epoch_num=500, out_model='retrained_net.model')
        print(acc, macs)

        # # otherwise (in the case where we do not have a log file.)
        # temp = CNN_train('haze1', validation=False, verbose=True, imgSize=128, batchsize=16)
        # cgp = [['input', 0], ['S_SumConvBlock_64_3', 0], ['S_ConvBlock_64_5', 1], ['S_SumConvBlock_128_1', 2], ['S_SumConvBlock_64_1', 3], ['S_SumConvBlock_64_5', 4], ['S_DeConvBlock_3_3', 5]]
        # acc = temp(cgp, 0, epoch_num=500, out_model='retrained_net.model')

    elif args.mode == 'reevolution':
        # restart evolution
        print('Restart Evolution')
        imgSize = 32
        with open(os.path.join(args.log_path, 'network_info.pkl'), mode='rb') as f:
            network_info = pickle.load(f)
        eval_f = CNNEvaluation(gpu_num=args.gpu_num, dataset='cifar10', reduced=args.reduced, verbose=True, epoch_num=50, batchsize=128,
                               imgSize=imgSize)
        cgp = CGP(network_info, eval_f, lam=args.lam, imgSize=imgSize, bias=args.bias)

        data = pd.read_csv(os.path.join(args.log_path, 'log_cgp.txt'), header=None)
        cgp.load_log(list(data.tail(1).values.flatten().astype(int)))
        cgp.modified_evolution(max_eval=250, mutation_rate=0.1, log_path=args.log_path)

    else:
        print('Undefined mode. Please check the "-m evolution or retrain or reevolution" ')
