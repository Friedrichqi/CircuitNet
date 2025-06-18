# Copyright 2022 CircuitNet. All rights reserved.

import argparse
import os
import sys

sys.path.append(os.getcwd())


class Parser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--task', default='congestion_gpdl')

        self.parser.add_argument('--save_path', default='work_dir/congestion_gpdl/')
        
        congestion_pretrained_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'congestion.pth')
        self.parser.add_argument('--pretrained', default=congestion_pretrained_path)

        self.parser.add_argument('--max_iters', default=400000)
        self.parser.add_argument('--period', default=[200000, 200000])
        self.parser.add_argument('--restart_factor', default=[1, 0.1])
        self.parser.add_argument('--plot_roc', action='store_true')
        self.parser.add_argument('--arg_file', default=None)
        self.parser.add_argument('--cpu', default=False)
        self.get_remainder()
    def get_remainder(self):
        training_set_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training_set')
        if self.parser.parse_known_args()[0].task == 'congestion_gpdl':
            self.parser.add_argument('--dataroot', default=os.path.join(training_set_path, 'congestion'))
            self.parser.add_argument('--ann_file_train', default='./files/train_N28.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test_N28.csv')
            self.parser.add_argument('--dataset_type', default='CongestionDataset')
            self.parser.add_argument('--batch_size', default=40)
            self.parser.add_argument('--aug_pipeline', default=['Flip'])
            
            self.parser.add_argument('--model_type', default='GPDL')
            self.parser.add_argument('--in_channels', default=3)
            self.parser.add_argument('--out_channels', default=1)
            self.parser.add_argument('--lr', default=5e-4)
            self.parser.add_argument('--min_lr', default=1e-8)
            self.parser.add_argument('--weight_decay', default=0)
            self.parser.add_argument('--loss_type', default='MSELoss')
            self.parser.add_argument('--eval-metric', default=['NRMS', 'SSIM', 'EMD'])

        elif self.parser.parse_known_args()[0].task == 'drc_routenet':
            self.parser.add_argument('--dataroot', default=os.path.join(training_set_path, 'DRC'))
            self.parser.add_argument('--ann_file_train', default='./files/train_N28.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test_N28.csv')
            self.parser.add_argument('--dataset_type', default='DRCDataset')
            self.parser.add_argument('--batch_size', default=8)
            self.parser.add_argument('--aug_pipeline', default=['Flip'])

            self.parser.add_argument('--model_type', default='RouteNet')
            self.parser.add_argument('--in_channels', default=9)
            self.parser.add_argument('--out_channels', default=1)
            self.parser.add_argument('--lr', default=2e-4)
            self.parser.add_argument('--weight_decay', default=1e-4)
            self.parser.add_argument('--loss_type', default='MSELoss')
            self.parser.add_argument('--eval-metric', default=['NRMS', 'SSIM'])
            self.parser.add_argument('--threshold', default=0.1)


        elif self.parser.parse_known_args()[0].task == 'irdrop_mavi':
            self.parser.add_argument('--dataroot', default=os.path.join(training_set_path, 'IR_drop'))
            self.parser.add_argument('--ann_file_train', default='./files/train_N28.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test_N28.csv')
            self.parser.add_argument('--dataset_type', default='IRDropDataset')
            self.parser.add_argument('--batch_size', default=2)

            self.parser.add_argument('--model_type', default='MAVI')
            self.parser.add_argument('--in_channels', default=1)
            self.parser.add_argument('--out_channels', default=4)
            self.parser.add_argument('--lr', default=2e-4)
            self.parser.add_argument('--weight_decay', default=1e-2)
            self.parser.add_argument('--loss_type', default='L1Loss')
            self.parser.add_argument('--eval_metric', default=['NRMS', 'SSIM'])
            self.parser.add_argument('--threshold', default=0.9885) # 5% after log

        else:
            raise ValueError
