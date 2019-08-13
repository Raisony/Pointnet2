from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F


NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8 # one cluster for each type
NUM_OBJECT_POINT = 512
g_type2class={'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3,
              'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}
g_class2type = {g_type2class[t]:t for t in g_type2class}
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
g_type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                    'Van': np.array([5.06763659,1.9007158,2.20532825]),
                    'Truck': np.array([10.13586957,2.58549199,3.2520595]),
                    'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                    'Person_sitting': np.array([0.80057803,0.5983815,1.27450867]),
                    'Cyclist': np.array([1.76282397,0.59706367,1.73698127]),
                    'Tram': np.array([16.17150617,2.53246914,3.53079012]),
                    'Misc': np.array([3.64300781,1.54298177,1.92320313])}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i,:] = g_type_mean_size[g_class2type[i]]


class PointNetfeat(nn.Module):
    def __init__(self, in_plane=4, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_plane, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x



class PointNet(nn.Module):
    def __init__(self, in_plane=4):
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_plane, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1 = torch.nn.Linear(512, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 3 + NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER*4)

        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)


    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        #x = x.view(-1, 1024)
        return x


class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = self.mp1(x)
        #print(x.size())
        x,_ = torch.max(x, 2)
        #print(x.size())
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class TNet(nn.Module):
    def __init__(self, num_seg_points=512):
        super(TNet, self).__init__()

        self.num_seg_points = num_seg_points

        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.max_pool = nn.MaxPool1d(num_seg_points)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        # (x has shape (batch_size, 3, num_seg_points))

        out = F.relu(self.conv1(x)) # (shape: (batch_size, 128, num_seg_points))
        out = F.relu(self.conv2(out)) # (shape: (batch_size, 256, num_seg_points))
        out = F.relu(self.conv3(out)) # (shape: (batch_size, 512, num_seg_points))
        out = self.max_pool(out) # (shape: (batch_size, 512, 1))
        out = out.view(-1, 512) # (shape: (batch_size, 512))

        out = F.relu(self.fc1(out)) # (shape: (batch_size, 256))
        out = F.relu(self.fc2(out)) # (shape: (batch_size, 128))

        out = self.fc3(out) # (shape: (batch_size, 3))

        return out

class BboxNet2(nn.Module):
    def __init__(self, num_seg_points=512, num_theta_bin=12, num_class=9):
        super(BboxNet2, self).__init__()

        self.num_seg_points = num_seg_points

        self.conv1 = nn.Conv1d(4+128, 256, 1)
        self.conv2 = nn.Conv1d(256, 512, 1)
        self.max_pool = nn.MaxPool1d(num_seg_points)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + num_theta_bin*2 + num_class*4)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        # (x has shape (batch_size, 3, num_seg_points))

        out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, 256, num_seg_points))
        out = F.relu(self.bn2(self.conv2(out))) # (shape: (batch_size, 512, num_seg_points))
        out = self.max_pool(out) # (shape: (batch_size, 512, 1))
        out = out.view(-1, 512) # (shape: (batch_size, 512))

        out = F.relu(self.fc1(out)) # (shape: (batch_size, 512))
        out = F.relu(self.fc2(out)) # (shape: (batch_size, 256))
        out = self.fc3(out) # (shape: (batch_size, 3 + 3 + 2*NH))

        return out

class BboxNet(nn.Module):
    def __init__(self, num_seg_points=512, num_theta_bin=12, num_class=9):
        super(BboxNet, self).__init__()

        self.num_seg_points = num_seg_points

        self.conv1 = nn.Conv1d(4, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.max_pool = nn.MaxPool1d(num_seg_points)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + num_theta_bin*2 + num_class*4)

    def forward(self, x):
        # (x has shape (batch_size, 3, num_seg_points))

        out = F.relu(self.conv1(x)) # (shape: (batch_size, 128, num_seg_points))
        out = F.relu(self.conv2(out)) # (shape: (batch_size, 128, num_seg_points))
        out = F.relu(self.conv3(out)) # (shape: (batch_size, 256, num_seg_points))
        out = F.relu(self.conv4(out)) # (shape: (batch_size, 512, num_seg_points))
        out = self.max_pool(out) # (shape: (batch_size, 512, 1))
        out = out.view(-1, 512) # (shape: (batch_size, 512))

        out = F.relu(self.fc1(out)) # (shape: (batch_size, 512))
        out = F.relu(self.fc2(out)) # (shape: (batch_size, 256))

        out = self.fc3(out) # (shape: (batch_size, 3 + 3 + 2*NH))

        return out




class InstanceSeg(nn.Module):
    def __init__(self, num_points=20000):
        super(InstanceSeg, self).__init__()

        self.num_points = num_points

        self.conv1 = nn.Conv1d(4, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.conv6 = nn.Conv1d(1088, 512, 1)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.conv8 = nn.Conv1d(256, 128, 1)
        self.conv9 = nn.Conv1d(128, 128, 1)
        self.conv10 = nn.Conv1d(128, 1, 1)
        self.max_pool = nn.MaxPool1d(num_points)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        batch_size = x.size()[0] # (x has shape (batch_size, 4, num_points))

        out = F.relu(self.conv1(x)) # (shape: (batch_size, 64, num_points))
        out = F.relu(self.conv2(out)) # (shape: (batch_size, 64, num_points))
        point_features = out

        out = F.relu(self.conv3(out)) # (shape: (batch_size, 64, num_points))
        out = F.relu(self.conv4(out)) # (shape: (batch_size, 128, num_points))
        out = F.relu(self.conv5(out)) # (shape: (batch_size, 1024, num_points))
        global_feature = self.max_pool(out) # (shape: (batch_size, 1024, 1))

        global_feature_repeated = global_feature.repeat(1, 1, self.num_points) # (shape: (batch_size, 1024, num_points))
        out = torch.cat([global_feature_repeated, point_features], 1) # (shape: (batch_size, 1024+64=1088, num_points))

        out = F.relu(self.conv6(out)) # (shape: (batch_size, 512, num_points))
        out = F.relu(self.conv7(out)) # (shape: (batch_size, 256, num_points))
        out = F.relu(self.conv8(out)) # (shape: (batch_size, 128, num_points))
        out = F.relu(self.conv9(out)) # (shape: (batch_size, 128, num_points))

        out = self.conv10(out) # (shape: (batch_size, 2, num_points))

        out = out.transpose(2,1).contiguous() # (shape: (batch_size, num_points, 2))
        out = F.log_softmax(out.view(-1, 1), dim=1) # (shape: (batch_size*num_points, 2))
        out = out.view(batch_size, self.num_points, 1) # (shape: (batch_size, num_points, 2))

        return out

    def get_loss(self, pred_seg, trg_seg):
        print(pred_seg.size(), trg_seg.size())
        return self.criterion(pred_seg, trg_seg)