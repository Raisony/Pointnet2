from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import sys
sys.path.append('../')
sys.path.append('./')
import torch
import torch.nn as nn
from collections import namedtuple
from utils.torch_utils import BinaryFocalLoss, SigmoidFocalClassificationLoss
from .pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from .pointnet2_lib.pointnet2 import pytorch_utils as pt_utils

def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])

    def model_fn(model, data, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)

            preds = model(inputs)
            loss = criterion(preds.view(labels.numel(), -1), labels.view(-1))

            _, classes = torch.max(preds, -1)
            acc = (classes == labels).float().sum() / labels.numel()

        return ModelReturn(preds, loss, {"acc": acc.item(), "loss": loss.item()})

    return model_fn


class Pointnet2MSG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """
    def __init__(self, num_classes=1, input_channels=6, use_xyz=True):
        super(Pointnet2MSG, self).__init__()
        self.name = 'Segmentation Net'
        self.criterion = SigmoidFocalClassificationLoss(alpha=0.25, gamma=2)
        self.sigmoid = nn.Sigmoid()
        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=4096,
                radii=[0.1, 0.5],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.5, 1.0],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[1.0, 2.0],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[2.0, 4.0],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=use_xyz,
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        seg_layers = []
        seg_layers.append(pt_utils.Conv1d(128, 128))
        seg_layers.append(nn.Dropout(0.5))
        seg_layers.append(pt_utils.Conv1d(128, num_classes, activation=None))
        self.FC_layer = nn.Sequential(*seg_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            #print(l_xyz[i].size(), l_features[i].size())
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
        global_feature = l_features[0].transpose(1,2).contiguous()
        #print(global_feature.size())
        return global_feature, self.FC_layer(l_features[0]).transpose(1, 2).contiguous()


    def get_loss(self, pred, trg):
        #print(self.sigmoid(pred).size(), trg.size())
        cls_target = (trg > 0).float()
        pos = (trg > 0).float()
        neg = (trg == 0).float()
        cls_weights = pos + neg
        pos_normalizer = pos.sum()
        cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)
        loss_cls = self.criterion(pred, cls_target, cls_weights)
        loss_cls_pos = (loss_cls * pos).sum()
        loss_cls_neg = (loss_cls * neg).sum()
        loss_cls = loss_cls.sum()
        return loss_cls

    def load_weights(self, base_file):
        import os
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict for {}: {}'.format(self.name, base_file))
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



class Pointnet2Bbox(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """
    def __init__(self, num_seg_points=512, num_theta_bin=12, num_class=4, input_channels=6, use_xyz=True, add_img_feat=False):
        super(Pointnet2Bbox, self).__init__()
        self.name = 'Box Net'
        self.input_channel = 4
        XYZ_UP_LAYER = [128, 128]
        NUM_POINTS= num_seg_points
        NPOINTS=[128, 32, -1]
        RADIUS=[0.2, 0.4, 100]
        NSAMPLE=[64, 64, 64]
        MLPS= [[128, 128, 128],
               [128, 128, 256],
               [256, 256, 512]]
        self.SA_modules = nn.ModuleList()

        c_in = 128
        for k in range(NPOINTS.__len__()):
            mlps = [c_in] + MLPS[k]
            npoint = NPOINTS[k] if NPOINTS[k] != -1 else None
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=npoint,
                    radii=[RADIUS[k]],
                    nsamples=[NSAMPLE[k]],
                    mlps=[mlps],
                    use_xyz=use_xyz,
                    bn=False
                )
            )
            c_in = mlps[-1]
            #print(c_in)

        pre_channel = mlps[-1]
        self.xyz_up_layer = pt_utils.SharedMLP([self.input_channel] + XYZ_UP_LAYER, bn=False)
        c_out = XYZ_UP_LAYER[-1]
        if add_img_feat:
            self.merge_down_layer = pt_utils.SharedMLP([c_out * 2 + c_out*3, c_out], bn=False)
        else:
            self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=False)

        reg_layers = []

        REG_FC = [256, 256]
        DP_RATIO = 0.1
        for k in range(0, REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, REG_FC[k], bn=False))
            pre_channel = REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, 3 + num_theta_bin*2 + num_class*4, activation=None))  # add a unary
        if DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, semantic_feature, img_feature):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz_input = pointcloud[..., 0:4].transpose(1, 2).unsqueeze(dim=3).contiguous()
        numpoints = pointcloud.size(1)
        #print(xyz_input.size())
        xyz_feature = self.xyz_up_layer(xyz_input)

        sem_feature = semantic_feature.transpose(1, 2).unsqueeze(dim=3).contiguous()

        img_feature_ext = img_feature.unsqueeze(1).repeat(1,numpoints,1).transpose(1,2).unsqueeze(dim=3).contiguous()
        #print(img_feature_ext.size(), sem_feature.size())  #torch.Size([100, 384, 512, 1]) torch.Size([100, 128, 512, 1])
        merged_feature = torch.cat((xyz_feature, sem_feature, img_feature_ext), dim=1)
        #merged_feature = torch.cat((xyz_feature, sem_feature), dim=1)
        #print(merged_feature.size())

        #print(xyz_feature.size(), merged_feature.size())  # torch.Size([100, 640, 512, 1])
        merged_feature = self.merge_down_layer(merged_feature)
        #print(merged_feature.size())
        l_xyz, l_features = [pointcloud[...,0:3].contiguous()], [merged_feature.squeeze(dim=3)]

        for i in range(len(self.SA_modules)):
            #print(l_xyz[i].size(), l_features[i].size())
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        box_reg = self.reg_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        return box_reg

