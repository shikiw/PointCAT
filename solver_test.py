# -*- coding: utf-8 -*-

import os
import sys
import datetime
# import logging
import shutil
import importlib
import numpy as np
from pathlib import Path

import torch
import torch.optim
import torchvision.utils as vutils
from torch.autograd import Variable

from tqdm import tqdm
import utils.provider as provider
from utils.utils import set_seed
from copy import deepcopy


from utils.logging import Logging_str
from utils.utils import AverageMeter, save_checkpoint
from model.networks import AutoEncoder, ProjHead #, Generator, Discriminator

from baselines import *

# include other paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model/classifier'))


class PointTester(object):
    def __init__(self, args):
        self.args = args

        ### Create Log Path ###
        self.log_path = os.path.join('./log/', self.args.experiment_dir)
        self.logfile_path = os.path.join(self.log_path, 'log_info.txt')
        self.log_string = Logging_str(self.logfile_path)

        ### Initialize Settings ###
        set_seed(2022)

        if self.args.dataset == 'ModelNet40':
            self.num_class = 40
        elif self.args.dataset == 'ShapeNetPart':
            self.num_class = 16
        else:
            raise NotImplementedError

        ### Initialize Model and Weights ###
        self.load_failed = False
        self.build_models()
        self.load_weights()

        ### Set Mode and Pre Head ###
        if args.use_pre_defense:
            self.set_pre_head(args.pre_defense)
        if not self.args.mode == 'test_ba':
            self.source_classifier = deepcopy(self.classifier)
            self.source_classifier.use_pre_defense = False



    def build_models(self):
        """Build new models for training.
        """
        MODEL = importlib.import_module(self.args.defended_model)
        classifier = MODEL.get_model(
            self.num_class, 
            normal_channel=self.args.normal,
            use_pre_defense=self.args.use_pre_defense
        )
        if self.args.mode == 'test_ba':
            MODEL = importlib.import_module(self.args.source_model)
            source_classifier = MODEL.get_model(
                self.num_class, 
                normal_channel=self.args.normal
            )
        noise_generator = AutoEncoder(
            k=self.num_class,
            input_point_nums=self.args.input_point_nums, 
            decoder_type=self.args.decoder_type, 
            args=self.args
        )

        # use multiple gpus
        if self.args.use_multi_gpu:
            classifier = nn.DataParallel(classifier)
            if self.args.mode == 'test_ba':
                source_classifier = nn.DataParallel(source_classifier)
            noise_generator = nn.DataParallel(noise_generator)
        
        self.classifier = classifier.cuda()
        if self.args.mode == 'test_ba':
            self.source_classifier = source_classifier.cuda()
        self.noise_generator = noise_generator.cuda()



    def load_weights(self):
        """Load weights from checkpoints.
        """
        try:
            checkpoint = torch.load(self.args.checkpoint_dir)
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            self.log_string.write('Use pretrained classifier')
        except:
            self.load_failed = True

        if self.args.mode == 'test_ba':
            try:
                if self.args.source_model_wo_defense:
                    if self.args.defended_model == 'dgcnn':
                        checkpoint = torch.load('../dgcnn/pytorch/pretrained/model.1024.t7')
                        self.source_classifier = nn.DataParallel(self.source_classifier)
                        self.source_classifier.load_state_dict(checkpoint)
                    elif self.args.defended_model == 'curvenet':
                        checkpoint = torch.load('../CurveNet/core/checkpoints/cls/model.t7')
                        self.source_classifier = nn.DataParallel(self.source_classifier)
                        self.source_classifier.load_state_dict(checkpoint)
                    else:
                        checkpoint = torch.load('../../Pointnet_Pointnet2_pytorch/log/classification/' \
                            + self.args.defended_model + '_'+ self.args.dataset.lower() \
                            + '/checkpoints/model_154.pth')
                        self.source_classifier.load_state_dict(checkpoint['model_state_dict'])
                else:
                    checkpoint = torch.load(self.args.source_model_dir)
                    self.source_classifier.load_state_dict(checkpoint['model_state_dict'])
            except:
                self.load_failed = True

        try:
            # checkpoint = torch.load('./log/pn_test_01/checkpoints/best-ng.pth')
            # checkpoint = torch.load('./log/dgcnn_test_01/checkpoints/best-ng.pth')
            self.start_epoch_ng = checkpoint['epoch']
            self.noise_generator.load_state_dict(checkpoint['model_state_dict'])
            self.log_string.write('Use pretrained noise-generator')
        except:
            self.start_epoch_ng = 0


    def set_pre_head(self, mode):
        """Support the following running mode:
        - sor:
        - srs: 
        - dupnet: point cloud defense method DUP-Net.
        - upsampling: point cloud upsampling.
        - denoising: point cloud denoising.
        """
        if mode == 'sor':
            pre_head = SORDefense(k=2, alpha=1.1)
        elif mode == 'srs':
            pre_head = SRSDefense(drop_num=500)
        elif mode == 'dupnet':
            pre_head = DUPNet(sor_k=2, sor_alpha=1.1, npoint=1024, up_ratio=4)
        elif mode == 'upsampling':
            pre_head = PUNet()
        elif mode == 'denoising':
            pre_head = PCNet()
            # pre_head = DMR()
        else:
            raise NotImplementedError

        if 0 and self.args.defended_model in ['dgcnn', 'curvenet']:
            self.classifier.module.set_pre_head(pre_head)
        else:
            self.classifier.set_pre_head(pre_head)


    def set_target(self, pc, target):
        """Set labels and original examples from dataset.
        """
        self.pc_ori = pc # [B, C, N]
        self.target = target.long() # [B]


    def run(self, mode):
        """Support the following running mode:
        - test: test the accuracy and the robustness of the given classifier.
        """
        if mode == 'test':
            self.run_test()
        elif mode == 'test_ba':
            self.run_test_ba()
        else:
            raise NotImplementedError


    def run_test_ng(self):
        """Evaluate the performance and robustness of CAT trained classifier.
        """
        self.classifier.eval()
        self.noise_generator.eval()
        batch_size = self.target.size(0)
        ori_input = self.pc_ori.clone().detach().transpose(1, 2).contiguous() # [B, N, C]
        ori_target = self.target # [B]

        ############################################
        ### (1) adv acc
        ############################################
        with torch.no_grad():
            pert = self.noise_generator(self.pc_ori, self.target)
            print(pert.mean())
            norm = torch.sum(pert ** 2, dim=[1, 2]) ** 0.5
            pert = pert / (norm[:, None, None] + 1e-9)
            pert = pert * np.sqrt(self.pc_ori.size(1) * self.pc_ori.size(2))
            print(pert.mean())
            pert = pert * self.args.eps
            print(pert.mean())
            assert 1==2
            self.pc_adv = torch.clamp(self.pc_ori+pert, min=-1, max=1)
            _, logits = self.classifier(self.pc_adv)
            pred = torch.argmax(logits, dim=-1)
            acc_num = (pred == ori_target).sum().item()
        self.acc_clean.update(val=acc_num, n=batch_size)



    def run_test(self):
        """Evaluate the performance and robustness of CAT trained classifier.
        """
        self.classifier.eval()
        self.source_classifier.eval()
        batch_size = self.target.size(0)
        ori_input = self.pc_ori.clone().detach().transpose(1, 2).contiguous() # [B, N, C]
        ori_target = self.target # [B]

        ############################################
        ### (1) clean acc
        ############################################
        with torch.no_grad():
            _, logits = self.classifier(self.pc_ori)
            pred = torch.argmax(logits, dim=-1)
            acc_num = (pred == ori_target).sum().item()
        self.acc_clean.update(val=acc_num, n=batch_size)


        ############################################
        ### (2) noisy and random dropping attacks
        ############################################

        # Random Noisy Attack
        noisy_attack = JitterAttack(self.classifier, sigma=0.04, clip=0.16)
        _, acc_num = noisy_attack(ori_input.detach(), ori_target)
        self.acc_noisy_1.update(val=acc_num, n=batch_size)

        noisy_attack = JitterAttack(self.classifier, sigma=0.08, clip=0.32)
        _, acc_num = noisy_attack(ori_input.detach(), ori_target)
        self.acc_noisy_2.update(val=acc_num, n=batch_size)

        # Random Dropping Attack
        drop_attack = DropAttack(self.classifier, drop_num=700)
        _, acc_num = drop_attack(ori_input.detach(), ori_target)
        self.acc_drop_1.update(val=acc_num, n=batch_size)

        drop_attack = DropAttack(self.classifier, drop_num=800)
        _, acc_num = drop_attack(ori_input.detach(), ori_target)
        self.acc_drop_2.update(val=acc_num, n=batch_size)


        ############################################
        ### (3) untargeted adversarial attacks
        ############################################

        # setup attack settings
        # budget, step_size, number of iteration
        # settings adopted from CVPR'20 paper GvG-P
        delta = 0.02
        budget = delta * np.sqrt(self.args.input_point_nums * 3)  # \delta * \sqrt(N * d)
        num_iter = 10
        step_size = budget / float(num_iter)

        # which adv_func to use?
        if self.args.adv_func == 'logits':
            adv_func = LogitsAdvLoss(kappa=self.args.kappa, mode='untargeted')
        else:
            adv_func = CrossEntropyAdvLoss(mode='untargeted')

        # which clip_func to use?
        clip_func = ClipPointsL2(budget=budget)

        # which dist_func to use?
        dist_func = L2Dist()
        # dist_func = ChamferDist()

        # FGM
        fgm_attack = FGM(self.source_classifier, self.classifier, adv_func=adv_func, budget=budget, dist_metric='l2')
        _, success_num = fgm_attack.attack(ori_input.detach(), ori_target, mode='untargeted')
        self.success_fgm_untar.update(val=success_num, n=batch_size)

        # I-FGM
        ifgm_attack = IFGM(self.source_classifier, self.classifier, adv_func=adv_func, clip_func=clip_func, budget=budget,
                                 step_size=step_size, num_iter=num_iter, dist_metric='l2')
        _, success_num = ifgm_attack.attack(ori_input.detach(), ori_target, mode='untargeted')
        self.success_ifgm_untar.update(val=success_num, n=batch_size)

        # MI-FGM
        mifgm_attack = MIFGM(self.source_classifier, self.classifier, adv_func=adv_func, clip_func=clip_func, budget=budget,
                                   step_size=step_size, num_iter=num_iter, mu=self.args.mu, dist_metric='l2')
        _, success_num = mifgm_attack.attack(ori_input.detach(), ori_target, mode='untargeted')
        self.success_mifgm_untar.update(val=success_num, n=batch_size)

        # PGD
        pgd_attack = PGD(self.source_classifier, self.classifier, adv_func=adv_func, clip_func=clip_func, budget=budget,
                               step_size=step_size, num_iter=num_iter, dist_metric='l2')
        _, success_num = pgd_attack.attack(ori_input.detach(), ori_target, mode='untargeted')
        self.success_pgd_untar.update(val=success_num, n=batch_size)

        # C&W
        cw_attack = CWPerturb(self.source_classifier, self.classifier, adv_func=adv_func, dist_func=dist_func,
                                    attack_lr=3e-3, init_weight=10., max_weight=80.,
                                    binary_step=5, num_iter=self.args.num_iter_cw)
        _, _, success_num = cw_attack.attack(ori_input.detach(), ori_target, mode='untargeted')
        self.success_cw_untar.update(val=success_num, n=batch_size)


        ############################################
        ### (4) targeted adversarial attacks
        ############################################

        # setup attack settings
        # budget, step_size, number of iteration
        # settings adopted from CVPR'20 paper GvG-P
        delta = self.args.budget
        budget = self.args.budget * np.sqrt(self.args.input_point_nums * 3)  # \delta * \sqrt(N * d)
        num_iter = int(self.args.num_iter)
        step_size = budget / float(num_iter)

        # generate the target label
        adv_target = []
        all_label_idx = [i for i in range(self.num_class)]
        for i in range(batch_size):
            idx = all_label_idx.copy()
            idx.remove(int(ori_target[i].cpu().item()))
            adv_target.append(np.random.choice(idx))
        assert len(adv_target) == batch_size
        adv_target = torch.Tensor(adv_target).to(ori_target.device) # [B]

        # which adv_func to use?
        if self.args.adv_func == 'logits':
            adv_func = LogitsAdvLoss(kappa=self.args.kappa, mode='targeted')
        else:
            adv_func = CrossEntropyAdvLoss(mode='targeted')

        # FGM
        fgm_attack = FGM(self.source_classifier, self.classifier, adv_func=adv_func, budget=budget, dist_metric='l2')
        _, success_num = fgm_attack.attack(ori_input.detach(), adv_target, mode='targeted')
        self.success_fgm_tar.update(val=success_num, n=batch_size)

        # I-FGM
        ifgm_attack = IFGM(self.source_classifier, self.classifier, adv_func=adv_func, clip_func=clip_func, budget=budget,
                                 step_size=step_size, num_iter=num_iter, dist_metric='l2')
        _, success_num = ifgm_attack.attack(ori_input.detach(), adv_target, mode='targeted')
        self.success_ifgm_tar.update(val=success_num, n=batch_size)

        # MI-FGM
        mifgm_attack = MIFGM(self.source_classifier, self.classifier, adv_func=adv_func, clip_func=clip_func, budget=budget,
                                   step_size=step_size, num_iter=num_iter, mu=self.args.mu, dist_metric='l2')
        _, success_num = mifgm_attack.attack(ori_input.detach(), adv_target, mode='targeted')
        self.success_mifgm_tar.update(val=success_num, n=batch_size)

        # PGD
        pgd_attack = PGD(self.source_classifier, self.classifier, adv_func=adv_func, clip_func=clip_func, budget=budget,
                               step_size=step_size, num_iter=num_iter, dist_metric='l2')
        _, success_num = pgd_attack.attack(ori_input.detach(), adv_target, mode='targeted')
        self.success_pgd_tar.update(val=success_num, n=batch_size)

        # C&W
        cw_attack = CWPerturb(self.source_classifier, self.classifier, adv_func=adv_func, dist_func=dist_func,
                                    attack_lr=self.args.attack_lr, init_weight=10., max_weight=80.,
                                    binary_step=self.args.binary_step, num_iter=self.args.num_iter_cw)
        _, _, success_num = cw_attack.attack(ori_input.detach(), adv_target, mode='targeted')
        self.success_cw_tar.update(val=success_num, n=batch_size)


    def show_results(self, mode):
        """Show the robustness performance of the given classifier.
        Support the following implement mode:
        - init: create new counting variables.
        - print: show the performance results.
        """
        if mode == 'init':
            # without any attacks
            self.acc_clean = AverageMeter()

            # noisy and random dropping attacks
            self.acc_noisy_1 = AverageMeter()
            self.acc_noisy_2 = AverageMeter()
            self.acc_drop_1 = AverageMeter()
            self.acc_drop_2 = AverageMeter()

            # untargeted adversarial attacks
            self.success_fgm_untar = AverageMeter()
            self.success_ifgm_untar = AverageMeter()
            self.success_mifgm_untar = AverageMeter()
            self.success_pgd_untar = AverageMeter()
            self.success_cw_untar = AverageMeter()

            # targeted adversarial attacks
            self.success_fgm_tar = AverageMeter()
            self.success_ifgm_tar = AverageMeter()
            self.success_mifgm_tar = AverageMeter()
            self.success_pgd_tar = AverageMeter()
            self.success_cw_tar = AverageMeter()

        elif mode == 'print':
            # without any attacks
            self.log_string.write('\nWithout any attacks:')

            self.log_string.write('Clean ACC ({:d}/{:d}): {:.4f}'.format(
                self.acc_clean.sum, self.acc_clean.count, self.acc_clean.avg))

            # noisy and random dropping attacks
            self.log_string.write('\nInput preprocessing attacks:')

            self.log_string.write('Random noisy_1 ACC ({:d}/{:d}): {:.4f}'.format(
                self.acc_noisy_1.sum, self.acc_noisy_1.count, self.acc_noisy_1.avg))
            self.log_string.write('Random noisy_2 ACC ({:d}/{:d}): {:.4f}'.format(
                self.acc_noisy_2.sum, self.acc_noisy_2.count, self.acc_noisy_2.avg))
            self.log_string.write('Random dropping_1 ACC ({:d}/{:d}): {:.4f}'.format(
                self.acc_drop_1.sum, self.acc_drop_1.count, self.acc_drop_1.avg))
            self.log_string.write('Random dropping_2 ACC ({:d}/{:d}): {:.4f}'.format(
                self.acc_drop_2.sum, self.acc_drop_2.count, self.acc_drop_2.avg))

            # untargeted adversarial attacks
            self.log_string.write('\nUntargeted adversarial attacks:')

            self.log_string.write('FGM attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_fgm_untar.sum, self.success_fgm_untar.count, self.success_fgm_untar.avg))
            self.log_string.write('I-FGM attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_ifgm_untar.sum, self.success_ifgm_untar.count, self.success_ifgm_untar.avg))
            self.log_string.write('MI-FGM attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_mifgm_untar.sum, self.success_mifgm_untar.count, self.success_mifgm_untar.avg))
            self.log_string.write('PGD attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_pgd_untar.sum, self.success_pgd_untar.count, self.success_pgd_untar.avg))
            self.log_string.write('CW attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_cw_untar.sum, self.success_cw_untar.count, self.success_cw_untar.avg))

            # targeted adversarial attacks
            self.log_string.write('\nTargeted adversarial attacks:')

            self.log_string.write('FGM attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_fgm_tar.sum, self.success_fgm_tar.count, self.success_fgm_tar.avg))
            self.log_string.write('I-FGM attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_ifgm_tar.sum, self.success_ifgm_tar.count, self.success_ifgm_tar.avg))
            self.log_string.write('MI-FGM attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_mifgm_tar.sum, self.success_mifgm_tar.count, self.success_mifgm_tar.avg))
            self.log_string.write('PGD attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_pgd_tar.sum, self.success_pgd_tar.count, self.success_pgd_tar.avg))
            self.log_string.write('CW attack ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_cw_tar.sum, self.success_cw_tar.count, self.success_cw_tar.avg))
        
        else:
            raise NotImplementedError


    def run_test_ba(self):
        """Evaluate the performance and robustness of CAT trained classifier.
        """
        self.classifier.eval()
        self.source_classifier.eval()

        # setup attack settings
        # budget, step_size, number of iteration
        # settings adopted from CVPR'20 paper GvG-P
        delta = 0.02
        budget = delta * np.sqrt(self.args.input_point_nums * 3)  # \delta * \sqrt(N * d)
        num_iter = 10
        step_size = budget / float(num_iter)

        # which adv_func to use?
        if self.args.adv_func == 'logits':
            adv_func = LogitsAdvLoss(kappa=self.args.kappa, mode='untargeted')
        else:
            adv_func = CrossEntropyAdvLoss(mode='untargeted')

        # which clip_func to use?
        clip_func = ClipPointsL2(budget=budget)

        # which dist_func to use?
        dist_func = L2Dist()

        # PGD
        pgd_attack = PGD(self.source_classifier, self.classifier, adv_func=adv_func, clip_func=clip_func, budget=budget,
                               step_size=step_size, num_iter=num_iter, dist_metric='l2')
        _, success_num = pgd_attack.attack(ori_input.detach(), ori_target, mode='untargeted')
        self.success_pgd_untar.update(val=success_num, n=batch_size)


    def show_results_ba(self, mode):
        """Show the robustness performance of the given classifier.
        Support the following implement mode:
        - init: create new counting variables.
        - print: show the performance results.
        """
        if mode == 'init':
            # untargeted adversarial attacks
            self.success_pgd_untar = AverageMeter()
        elif mode == 'print':
            if self.args.source_model_wo_defense:
                print('Source: clean ', self.args.source_model)
            else:
                print('Source: defended ', self.args.source_model)
            print('Target: defended ', self.args.defended_model)
            self.log_string.write('Black-box PGD ASR ({:d}/{:d}): {:.4f}'.format(
                self.success_pgd_untar.sum, self.success_pgd_untar.count, self.success_pgd_untar.avg))
