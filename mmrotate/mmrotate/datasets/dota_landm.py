# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import zipfile
from collections import defaultdict
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS

from mmrotate.datasets import DOTADataset

@ROTATED_DATASETS.register_module()
class DOTALandmDataset(DOTADataset):
    """DOTA dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')

    PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):

        super(DOTALandmDataset, self).__init__(ann_file, pipeline,version, difficulty, **kwargs)

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.bmp')
            # ann_files = glob.glob(ann_folder + '/*.png')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                # img_name = img_id + '.png'
                img_name = img_id + '.bmp'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                # img_name = img_id + '.png'
                img_name = img_id + '.jpg'
                # img_name = img_id + '.bmp'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_landms = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []
                # f = open('/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-101-landmfliter-test/data.txt', 'a+')
                # f.write(str(img_name) + '\n')
                # print('img_name: ', img_name)
                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            # print('version: ', self.version)
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                            # print('poly: ', poly)
                            # print('bbox: ', [x, y, w, h, a])
                            # print('a: ', a)
                        except:  # noqa: E722
                            continue
                        cls_name = bbox_info[8]
                        difficulty = int(bbox_info[9])
                        label = cls_map[cls_name]
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)
                            # print('poly: ', poly)
                            # print('ploy[0:4]: ', poly[0:4].extend([x,y]))
                            # print('x: ', x)
                            # print('y: ', y)
                            gt_landms.append([poly[0], poly[1], poly[2], poly[3], x, y])# 添加左上右上和中心点坐标。
                            # print('gt_landms: ', gt_landms)
                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                    data_info['ann']['landms'] = np.array(
                        gt_landms, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)
                    data_info['ann']['landms'] = np.zeros((0, 6), dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['landm_fields'] = []
        # print('results_dataset: ', results)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            # print('data: ', data)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    