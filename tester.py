# -*- coding: utf-8 -*-

import os
import time
import random
import importlib
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.optim
import torchvision.utils as vutils

### arguments related
from arguments import Arguments

### data related
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from torch.utils.data import DataLoader, TensorDataset
import utils.provider as provider

### model related
from solver_test import PointTester
from loss import *

### others related ###
from utils.logging import Logging_str
from utils.utils import set_seed
from baselines import *




class Tester(object):

    def __init__(self, args):
        self.args = args

        ### Check and Create Log Path ###
        log_path = Path('./log/')
        log_path.mkdir(exist_ok=True)
        log_path = log_path.joinpath(self.args.experiment_dir)
        log_path.mkdir(exist_ok=True)
        self.log_path = log_path
        self.logfile_path = os.path.join(self.log_path, 'log_info.txt')
        self.log_string = Logging_str(self.logfile_path)

        ### Model Function Initialization ###
        self.model = PointTester(args)

        ### Dataset Preparation ###
        self.load_data()


    def load_data(self):
        """
        Load the data file. When running our code in the first time, we 
        intend to save .pt data file to acelerate the training procedure.
        """
        self.log_string.write('Start Loading Dataset...')

        # make directory
        data_pt_path = '../data/' + str(self.args.dataset) + '_' + str(self.args.input_point_nums) + '_pt/'
        if not os.path.isdir(data_pt_path):
            os.mkdir(data_pt_path)

        # make data file if not existing
        pt_path_test_points = os.path.join(data_pt_path, 'test_points.pt')
        pt_path_test_target = os.path.join(data_pt_path, 'test_target.pt')
        if not os.path.isfile(pt_path_test_points) or not os.path.isfile(pt_path_test_target):
            self.make_dataloader()
            self.save_data_as_pt(self.testDataLoader, data_pt_path, mode='test')

        # load the pt file
        points_test = torch.load(pt_path_test_points) # [M_te, N, C]
        target_test = torch.load(pt_path_test_target) # [M_te, 1]
        self.points_test = points_test.transpose(1, 2)
        self.target_test = target_test[:, 0].long()

        # set tensor loader
        self.make_dataloader(
            testset=tuple([points_test, target_test])
        )

        self.log_string.write('Finish Loading Dataset...')


    def make_dataloader(self, testset=None):
        """
        Load the dataloader for training or testing.
        """
        if testset != None:
            print('Loading from the given data tensors...')
            TEST_DATASET = TensorDataset(testset[0], testset[1])
        else:
            if self.args.dataset == 'ModelNet40':
                TEST_DATASET = ModelNetDataLoader(
                    root=self.args.data_path, 
                    npoint=self.args.input_point_nums, 
                    split='test', 
                    normal_channel=self.args.normal
                )
            elif self.args.dataset == 'ShapeNetPart':
                TEST_DATASET = PartNormalDataset(
                    root=self.args.data_path, 
                    npoints=self.args.input_point_nums, 
                    split='test', 
                    normal_channel=self.args.normal
                )
            else:
                raise NotImplementedError

        self.testDataLoader = torch.utils.data.DataLoader(
            TEST_DATASET, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            # pin_memory=True,
            num_workers=self.args.num_workers
        )


    def save_data_as_pt(self, dataloader, pt_path, mode='train'):
        """
        Turn the given dataloader to data tensors and save them.
        """
        for batch_id, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), smoothing=0.9):
            # prepare data for training
            points, target = data
            # concat to a big tensor
            if batch_id == 0:
                points_gather = points
                target_gather = target
            else:
                points_gather = torch.cat([points_gather, points], dim=0)
                target_gather = torch.cat([target_gather, target], dim=0)
        # get the path to save
        pt_path = pt_path + str(mode)
        points_path = pt_path + '_points.pt'
        target_path = pt_path + '_target.pt'
        # save the pt
        torch.save(points_gather, points_path)
        torch.save(target_gather, target_path)


    def data_preprocess(self, data, mode='train'):
        """
        Preprocess the given data and label.
        """
        points, target = data

        if mode == 'train':
            # pointcloud augmentation
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points) # [B, N, C]

        points = points.transpose(2, 1) # [B, C, N]
        target = target[:, 0] # [B]

        points = points.cuda()
        target = target.cuda()

        return points, target


    def run(self, mode):
        """Support the following running mode:
        - train: implement CAT on the prepared classifier.
        - test_normal: test the accuracy and the robustness of CAT trained classifiers 
        under point noisy, point dropping and white-box adversarial attacks.
        - test_aa: test the accuracy and the robustness of CAT trained classifiers 
        under auto-attack.
        - test_ba: test the accuracy and the robustness of CAT trained classifiers 
        under black-box attack.
        """
        if mode == 'test_normal':
            self.test_normal()
        elif mode == 'test_aa':
            self.test_aa()
        elif mode == 'test_ba':
            self.test_ba()
        else:
            raise NotImplementedError


    def test_normal(self):
        """Implement the test procedure for normal robustness.
        """
        if self.model.load_failed:
            ### if loading checkpoint failed
            #######################################
            raise NotImplementedError
        else:
            ### if loading checkpoint successfully
            #######################################
            # test performance
            self.log_string.write('Start Robustness Validation Step...')
            self.model.show_results(mode='init')
            for batch_id, data in tqdm(enumerate(self.testDataLoader), total=len(self.testDataLoader)):
                # prepare data for testing
                points, target = self.data_preprocess(data=data, mode='test')
                self.model.set_target(pc=points, target=target)

                # test on the batch data
                self.model.run(mode='test')

            # show performance
            self.model.show_results(mode='print')


    def test_aa(self):
        """Implement the test procedure for auto-attack robustness.
        """
        if self.model.load_failed:
            ### if loading checkpoint failed
            #######################################
            raise NotImplementedError
        else:
            ### if loading checkpoint successfully
            #######################################
            if args.defended_model == 'pointnet2_cls_msg':
                auto_attack = AutoAttack(
                    model=self.model.classifier.eval(), 
                    norm='L2', 
                    eps=0.3, 
                    seed=2022, 
                    version='rand', 
                    log_path='./baselines/auto_attack/a.txt'
                )
            else:
                auto_attack = AutoAttack(
                    model=self.model.classifier.eval(), 
                    norm='L2', 
                    eps=0.3, 
                    seed=2022, 
                    version='standard', 
                    log_path='./baselines/auto_attack/a.txt'
                )

            _, y_adv = auto_attack.run_standard_evaluation(
                x_orig=self.points_test,
                y_orig=self.target_test,
                bs=16, 
                return_labels=True
            )

            success_num = (y_adv != self.target_test).sum().item()
            asr = float(success_num) / self.target_test.size(0)
            print('Auto-attack ASR: ', asr)


    def test_ba(self):
        """Implement the test procedure for black-box robustness.
        """
        if self.model.load_failed:
            ### if loading checkpoint failed
            #######################################
            raise NotImplementedError
        else:
            ### if loading checkpoint successfully
            #######################################
            # test performance
            self.log_string.write('Start Robustness Validation Step...')
            self.model.show_results_ba(mode='init')
            for batch_id, data in tqdm(enumerate(self.testDataLoader), total=len(self.testDataLoader)):
                # prepare data for testing
                points, target = self.data_preprocess(data=data, mode='test')
                self.model.set_target(pc=points, target=target)

                # test on the batch data
                self.model.run(mode='test_ba')

            # show performance
            self.model.show_results_ba(mode='print')



if __name__ == "__main__":
    args = Arguments(stage='test').parser().parse_args()
    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    
    if not os.path.isdir('./log/'):
        os.mkdir('./log/')
    if not os.path.isdir('./data'):
        os.mkdir('./data')

    set_seed(2022)
    tester = Tester(args)
    tester.run(mode=args.mode)
    