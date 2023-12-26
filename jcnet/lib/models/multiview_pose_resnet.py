# ------------------------------------------------------------------------------
# multiview.pose3d.torch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class ChannelWiseFC(nn.Module):

    def __init__(self, size):
        super(ChannelWiseFC, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(size, size))   #将不可训练的Tensor变换为可训练的Parameter
        self.weight.data.uniform_(0, 0.1)

    def forward(self, input):
        N, C, H, W = input.size()
        input_reshape = input.reshape(N * C, H * W)
        output = torch.matmul(input_reshape, self.weight)     #矩阵乘法
        output_reshape = output.reshape(N, C, H, W)
        return output_reshape


class Aggregation(nn.Module):

    def __init__(self, cfg, weights=[0.4, 0.2, 0.2, 0.2]):
        super(Aggregation, self).__init__()
        NUM_NETS = 12
        size = int(cfg.NETWORK.HEATMAP_SIZE[0])    #热图大小，h36m 96，oc-per 64.
        self.weights = weights
        self.aggre = nn.ModuleList()             #类list
        for i in range(NUM_NETS):
            self.aggre.append(ChannelWiseFC(size * size))

    def sort_views(self, target, all_views):
        indicator = [target is item for item in all_views]  #当target在all_views中所处位置的值为True，其余为false
        new_views = [target.clone()]
        for i, item in zip(indicator, all_views):
            if not i:
                new_views.append(item.clone())
        return new_views

    def fuse_with_weights(self, views):
        target = torch.zeros_like(views[0])          #生成形状与views[0]一致的全零
        for v, w in zip(views, self.weights):
            target += v * w
        return target

    def forward(self, inputs):
        index = 0
        outputs = []
        nviews = len(inputs)
        for i in range(nviews):
            sorted_inputs = self.sort_views(inputs[i], inputs)
            warped = [sorted_inputs[0]]       #选出的视图
            for j in range(1, nviews):
                fc = self.aggre[index]
                fc_output = fc(sorted_inputs[j])
                warped.append(fc_output)
                index += 1
            output = self.fuse_with_weights(warped)      #warped选出的视图与其他视图的集合，下标为0的是当前选出的视图
            outputs.append(output)
        return outputs


class MultiViewPose(nn.Module):

    def __init__(self, PoseResNet, Aggre, CFG):
        super(MultiViewPose, self).__init__()
        self.config = CFG
        self.resnet = PoseResNet

    def forward(self, views):
        if isinstance(views, list):    #判断是否为多视图输入
            nviews = len(views)
            all_views_input = torch.cat(views, dim=0)       #torch.cat()拼接多个张量
            all_views_heatmaps, _ = self.resnet(all_views_input)    #各个视图热图总和张量
            single_views = torch.chunk(all_views_heatmaps, chunks=nviews, dim=0)    #分离单个视图各自的热图
            multi_views = []
            return single_views, multi_views
        else:           #非多视图输入
            return self.resnet(views)


def get_multiview_pose_net(resnet, CFG):
    model = MultiViewPose(resnet, None, CFG)
    return model

