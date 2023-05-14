"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from model_utils.pointconv_util import PointConvDensitySetAbstraction
from model.networks import ProjHead

class get_model(nn.Module):
    def __init__(self, k = 40, normal_channel=True, use_pre_defense=False):
        super(get_model, self).__init__()
        if normal_channel:
            feature_dim = 3
        else:
            feature_dim = 0
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=feature_dim + 3, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], bandwidth = 0.4, group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.7)
        self.fc3 = nn.Linear(256, k)

        self.use_pre_defense = use_pre_defense

    def set_pre_head(self, pre_defense_head):
        # if we need preprocess-based defense module
        assert pre_defense_head is not None
        self.pre_defense_head = pre_defense_head

    def forward(self, xyz, feat=None):
        # if it uses pre-defense
        if self.use_pre_defense:
            assert self.pre_defense_head is not None
            xyz = self.pre_defense_head(xyz)

        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))

        h = x
        # if it needs projection
        if isinstance(self.fc3, ProjHead):
            x = self.fc3(x)
        else:
            x = self.fc3(x)
            x = F.log_softmax(x, -1)
        return h, x



if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048))
    label = torch.randn(8,16)
    model = PointConvDensityClsSsg(num_classes=40)
    output= model(input)
    print(output.size())

