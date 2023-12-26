# ------------------------------------------------------------------------------
# multiview.pose3d.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append("../")

import cv2
import copy
import random
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import affine_transform_pts
from multiviews import cameras as cam_utils

downsample = 16

class JointsDataset(Dataset):

    def __init__(self, cfg, subset, is_train, transform=None):
        self.is_train = is_train
        self.subset = subset

        self.root = cfg.DATASET.ROOT
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        self.sigma = cfg.NETWORK.SIGMA
        self.transform = transform
        self.db = []

        self.num_joints = 20
        self.union_joints = {
            0: 'root',  # 根部
            1: 'rhip',  # 右臀部
            2: 'rkne',  # 右膝盖
            3: 'rank',  # 右脚踝
            4: 'lhip',  # 左臀部
            5: 'lkne',  # 左膝盖
            6: 'lank',  # 左脚踝
            7: 'belly',  # 腹部
            8: 'thorax', # 胸部
            9: 'neck',  # 颈部
            10: 'upper neck',
            11: 'nose',  # 鼻子
            12: 'head',  # 头部
            13: 'head top',
            14: 'lsho',  # 左肩
            15: 'lelb',  # 左手肘
            16: 'lwri',  # 左手腕
            17: 'rsho',  # �������������������������������������������������������������������肩
            18: 'relb',  # 右手肘
            19: 'rwri'  # 右手腕
        }
        self.actual_joints = {}
        self.u2a_mapping = {}

        _y, _x = torch.meshgrid(torch.arange(self.image_size[0] // downsample),
                                torch.arange(self.image_size[1] // downsample))
        grid = torch.stack([_x, _y], dim=-1)  # Tensor, size:(32, 32, 2) val: 0-32
        grid = grid * downsample + downsample / 2.0 - 0.5  # Tensor, size:(32, 32, 2), val: 0-256
        self.grid = grid.view(-1, 2)  # Tensor, size:(hw, 2), val: 0-256

        self.totalcapture_template_meta = dict()
        self.totalcapture_template_meta['joints_gt'] = np.zeros((16, 3))
        self.totalcapture_template_meta['bone_vec'] = dict()
        for bv in ['Head', 'Sternum', 'Pelvis', 'L_UpArm', 'R_UpArm', 'L_LowArm',
                   'R_LowArm', 'L_UpLeg', 'R_UpLeg', 'L_LowLeg', 'R_LowLeg']:
            self.totalcapture_template_meta['bone_vec'][bv] = np.zeros((3,))

        self.totalcapture_template_meta['camera'] = \
            {'R': np.zeros((3,3)), 'T': np.zeros((3,1)), 'fx': 0., 'fy': 0., 'cx': 0., 'cy': 0.,
             'distor': 0., 'k': np.zeros((3,1)), 'p': np.zeros((2,1)), 'name': 'null'}
        self.totalcapture_template_meta['bone_vectors'] = \
            {2: np.zeros((3,)), 3: np.zeros((3,)), 4: np.zeros((3,)), 5: np.zeros((3,)), 6: np.zeros((3,)),
             11: np.zeros((3,)), 12: np.zeros((3,)), 13: np.zeros((3,)), 14: np.zeros((3,)), 8: np.zeros((3,))}



    def get_mapping(self):
        union_keys = list(self.union_joints.keys())
        union_values = list(self.union_joints.values())

        mapping = {k: '*' for k in union_keys}
        for k, v in self.actual_joints.items():
            idx = union_values.index(v)
            key = union_keys[idx]
            mapping[key] = k
        return mapping

    def do_mapping(self):
        mapping = self.u2a_mapping
        for item in self.db:
            joints = item['joints_2d']
            joints_vis = item['joints_vis']

            njoints = len(mapping)
            joints_union = np.zeros(shape=(njoints, 2))
            joints_union_vis = np.zeros(shape=(njoints, 3))

            for i in range(njoints):
                if mapping[i] != '*':
                    index = int(mapping[i])
                    joints_union[i] = joints[index]
                    joints_union_vis[i] = joints_vis[index]
            item['joints_2d'] = joints_union
            item['joints_vis'] = joints_union_vis

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx, source='h36m', **kwargs):
        db_rec = copy.deepcopy(self.db[idx])
        image_dir = 'images.zip@' if self.data_format == 'zip' else ''
        image_file = osp.join(self.root, db_rec['source'], image_dir, 'images',
                              db_rec['image'])
        image_file = image_file.replace('\\', '/')


        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        
        joints = db_rec['joints_2d'].copy()
        joints_vis = db_rec['joints_vis'].copy()

        center = np.array(db_rec['center']).copy()
        scale = np.array(db_rec['scale']).copy()
        rotation = 0

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rotation = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

        trans = get_affine_transform(center, scale, rotation, self.image_size)
        # ! Notice: this trans represents full image to cropped image,
        # not full image->heatmap

        input = cv2.warpAffine(
            data_numpy,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                if (np.min(joints[i, :2]) < 0 or
                        joints[i, 0] >= self.image_size[0] or
                        joints[i, 1] >= self.image_size[1]):
                    joints_vis[i, :] = 0

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        # 3x3 data augmentation affine trans (scale rotate)
        # !!! Notice: this transformation contains both heatmap->image scale affine
        # and data augmentation affine
        aug_trans = np.eye(3, 3)
        aug_trans[0:2] = trans  # full img -> cropped img
        hm_scale = self.heatmap_size / self.image_size
        scale_trans = np.eye(3,3)  # cropped img -> heatmap
        scale_trans[0,0] = hm_scale[1]
        scale_trans[1, 1] = hm_scale[0]
        aug_trans = scale_trans @ aug_trans

        meta = {
            'scale': scale,
            'center': center,
            'rotation': rotation,
            'joints_2d': db_rec['joints_2d'],
            'joints_2d_transformed': joints,
            'joints_vis': joints_vis,
            'source': db_rec['source'],
            'heatmap_size': self.heatmap_size,
            'aug_trans': aug_trans,
        }
        if source == 'totalcapture':
            meta['joints_gt'] = db_rec['joints_gt']
            meta['camera'] = db_rec['camera']
            meta['action'] = db_rec['action']
            meta['subaction'] = db_rec['subaction']
            meta['subject'] = db_rec['subject']
            meta['image_id'] = db_rec['image_id']
        elif source in ['h36m']:
            meta['camera'] = db_rec['camera']
            meta['joints_gt'] = cam_utils.camera_to_world_frame(db_rec['joints_3d_camera'], db_rec['camera']['R'], db_rec['camera']['T'])
            camera = db_rec['camera'].copy()
            trans_inv= get_affine_transform(center, scale, rotation, self.image_size, inv=1)
            coords_ray = self.create_3d_ray_coords(camera, trans_inv)
            meta['rays'] = coords_ray
            cam_center = torch.Tensor(camera['T'].T)
            meta['cam_center'] = cam_center
            meta['action'] = db_rec['action']
            meta['subaction'] = db_rec['subaction']
            meta['subject'] = db_rec['subject']
            meta['image_id'] = db_rec['image_id']
            
        elif source == 'panoptic':
            meta['camera'] = db_rec['camera']
            meta['joints_gt'] = db_rec['joints_gt']
        elif source in ['unrealcv']:
            meta['camera'] = db_rec['camera']
            meta['joints_gt'] = db_rec['joints_gt']
        else:
            assert 0==1, 'No such dataset definition in JointDataset'
        return input, target, target_weight, meta

    def generate_target(self, joints_3d, joints_vis):
        target, weight = self.generate_heatmap(joints_3d, joints_vis)
        return target, weight

    def generate_heatmap(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros(
            (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))

            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def create_3d_ray_coords(self, camera, trans_inv):
        multiplier = 1.0                        # avoid numerical instability
        grid = self.grid.clone()                # Tensor,   (hw, 2), val in 0-256
        # transform to original image R.T.dot(x.T) + T
        coords = affine_transform_pts(grid.numpy(), trans_inv)  # array, size: (hw, 2), val: 0-1000

        coords[:, 0] = (coords[:, 0] - camera['cx'][0]) / camera['fx'][0] * multiplier      # array
        coords[:, 1] = (coords[:, 1] - camera['cy'][0]) / camera['fy'][0] * multiplier

        # (hw, 3) 3D points in cam coord
        coords_cam = np.concatenate((coords,
                                     multiplier * np.ones((coords.shape[0], 1))), axis=1)   # array

        coords_world = (camera['R'].T @ coords_cam.T + camera['T']).T  # (hw, 3)    in world coordinate    array
        coords_world = torch.from_numpy(coords_world).float()  # (hw, 3)
        return coords_world