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


import os.path as osp
import numpy as np
import pickle
import collections
from operator import itemgetter
import logging

from dataset.joints_dataset import JointsDataset
from multiviews import cameras as cam_utils


class TotalCaptureDataset(JointsDataset):

    def __init__(self, cfg, image_set, is_train, transform=None, selected_cam=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'lsho',
            11: 'lelb',
            12: 'lwri',
            13: 'rsho',
            14: 'relb',
            15: 'rwri'
        }
        # 0 'Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 6 'LeftFoot',
        # 7 'Spine', 'Neck', 'Head', 10 'LeftArm', 'LeftForeArm', 'LeftHand',
        # 13'RightArm', 'RightForeArm', 15'RightHand'

        # # load grouping and db from pickle to accelerate dataset loading
        # if selected_cam:
        #     self.selected_cam = selected_cam
        # else:
        #     self.selected_cam = [0, 2, 4, 6]  # list rather than tuple

        self.selected_cam = list(cfg.MULTI_CAMS.SELECTED_CAMS)

        selected_cam_string = ''.join([str(c+1) for c in self.selected_cam])
        self.u2a_mapping = super().get_mapping()
        grouping_db_pickle_file = osp.join(self.root, 'totalcapture', 'quickload',
                                           'totalcapture_quickload_{}_cam_{}.pkl'
                                           .format(image_set, selected_cam_string))
        if osp.isfile(grouping_db_pickle_file):
            with open(grouping_db_pickle_file, 'rb') as f:
                grouping_db = pickle.load(f)
                self.grouping = grouping_db['grouping']
                self.db = grouping_db['db']
        else:
            anno_file = osp.join(self.root, 'totalcapture', 'annot',
                                 'totalcapture_{}.pkl'.format(image_set))
            self.db = self.load_db(anno_file)
            super().do_mapping()
            self.grouping = self.get_group(self.db)
            grouping_db_to_dump = {'grouping': self.grouping, 'db': self.db}  # dump full group
            with open(grouping_db_pickle_file, 'wb') as f:
                pickle.dump(grouping_db_to_dump, f)
        if self.is_train:
            self.grouping = self.grouping[::1]
        else:
            self.grouping = self.grouping[::16]

        self.group_size = len(self.grouping)
        # use HumanBody.imubone as bone_vectors dict key
        self.imubone_mapping = {'Head': 8, 'Pelvis': 2, 'L_UpArm': 11, 'R_UpArm': 13, 'L_LowArm': 12, 'R_LowArm': 14,
                                'L_UpLeg': 5, 'R_UpLeg': 3, 'L_LowLeg': 6, 'R_LowLeg': 4}  # todo read from body
        # self.__getitem__(0)  # just make sure totalcapture_template_meta in JointDataset initialized
        # this will lead to zipfile.BadZipFile: Bad CRC-32 for file

    def index_to_action_names(self):
        return {
            1: 'rom',
            2: 'walking',
            3: 'acting',
            4: 'running',
            5: 'freestyle'
        }

    def filter_views_from_group(self):
        filtered_group = []
        for g in self.grouping:
            ng = itemgetter(*self.selected_cam)(g)
            filtered_group.append(ng)
        return filtered_group

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset

    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            camera_id = db[i]['camera_id']
            if keystr not in grouping:
                grouping[keystr] = [-1, -1, -1, -1, -1, -1, -1, -1]
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        return filtered_grouping

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        items = self.grouping[idx]
        for item in items:
            i, t, w, m = super().__getitem__(item, source='totalcapture')  # extra args for data source
            m['scale'] = m['scale'].astype(np.float32)
            m['center'] = m['center'].astype(np.float32)
            m['rotation'] = int(m['rotation'])
            m['action'] = int(m['action'])
            m['subaction'] = int(m['subaction'])
            m['subject'] = int(m['subject'])
            m['image_id'] = int(m['image_id'])             
            if 'name' in m['camera']:
                del m['camera']['name']
            for k in m['camera']:
                m['camera'][k] = m['camera'][k]
            
            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)

        return input, target, weight, meta

    def __len__(self):
        return self.group_size

    def get_key_str(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()
        nview = len(self.selected_cam)

        if 'threshold' in kwargs:
            threshold = kwargs['threshold']
        else:
            threshold = 0.0125
        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))

        gt = []
        flat_items = []
        box_lengthes = []
        for items in self.grouping:
            for item in items:
                gt.append(self.db[item]['joints_2d'][su, :2])
                flat_items.append(self.db[item])
                boxsize = np.array(self.db[item]['scale']).sum()*100.0  # crop img pixels
                box_lengthes.append(boxsize)
        gt = np.array(gt)
        if pred.shape[1] == 20:
            pred = pred[:, su, :2]
        elif pred.shape[1] == 16:
            pred = pred[:, :, :2]
        detection_threshold = np.array(box_lengthes).reshape((-1, 1)) * threshold

        distance = np.sqrt(np.sum((gt - pred)**2, axis=2))
        detected = (distance <= detection_threshold)

        # joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])
        # in many frames, people walk out of certain camera views
        joint_validity = self.get_joint_inview(gt, 1919, 1079)
        detected = np.logical_and(detected, joint_validity)
        num_valid_joint = np.sum(joint_validity, axis=0)
        joint_detection_rate = np.sum(detected, axis=0) / num_valid_joint  # shape (num_joint,)
        all_joint_detection_rate = np.sum(detected) / np.sum(num_valid_joint)
        # all detected joints / all inview joints

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = joint_detection_rate[i]
        # return name_values, np.mean(joint_detection_rate)
        detected_int = detected.astype(np.int)
        detected_int[joint_validity == False] = -1
        nsamples, njoints = detected.shape
        per_grouping_detected = detected_int.reshape(nsamples//nview, nview*njoints)
        # per_grouping_detected of shape (n_groupings, njoints*nview), -1: not-vis, 0: not detected, 1: detected
        return name_values, all_joint_detection_rate, per_grouping_detected

    def get_joint_inview(self, gt, box_width, box_height):
        """

        :param gt: shape of (num_samples, num_joints, 2)
        :param box_width: int  x_max
        :param box_height: int y_max
        :return: shape of (num_samples, num_joints) indicating if certain joint is not outside image
        """
        # y is 2nd cord of pose2d, and height, row of canvas
        is_y_in_canvas = np.logical_and(gt[:,:,1] >= 0., gt[:,:,1] <= box_height)
        is_x_in_canvas = np.logical_and(gt[:,:,0] >= 0., gt[:,:,0] <= box_width)
        is_pt_in_canvas = np.logical_and(is_x_in_canvas, is_y_in_canvas)
        return is_pt_in_canvas


    def evaluate_3d(self, preds3d, thresholds=None):
        if thresholds is None:
            thresholds = [5., 10., 15., 20., 25., 50., 75., 100., 125., 150.,]

        gt3d = []
        for idx, items in enumerate(self.grouping):

            db_rec = self.db[items[0]]
            j3d_global = cam_utils.camera_to_world_frame(db_rec['joints_3d_camera'], db_rec['camera']['R'], db_rec['camera']['T'])
            gt3d.append(j3d_global)
        gt3d = np.array(gt3d)

        assert preds3d.shape == gt3d.shape, 'shape mismatch of preds and gt'
        distance = np.sum((preds3d - gt3d)**2, axis=2)

        num_groupings = len(gt3d)
        pcks = []
        for thr in thresholds:
            detections = distance <= thr**2
            detections_perjoint = np.sum(detections, axis=0)
            pck_perjoint = detections_perjoint / num_groupings
            # pck_avg = np.average(pck_perjoint, axis=0)
            pcks.append(pck_perjoint)

        return thresholds, pcks