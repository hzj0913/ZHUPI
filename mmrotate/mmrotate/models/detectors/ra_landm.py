# Copyright (c) SJTU. All rights reserved.
# 旋转框 + 关键点检测
import warnings
import mmcv
import numpy as np
import torch
from mmcv.runner import ModuleList
from mmrotate.core import imshow_det_rbboxes_landms
from mmrotate.core import rbboxlandm2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .r3det import R3Det
from .utils import FeatureRefineModule


@ROTATED_DETECTORS.register_module()
class RALandm(R3Det):
    """Rotated Refinement RetinaNet."""

    def __init__(self,
                 num_refine_stages,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 frm_cfgs=None,
                 refine_heads=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RALandm, self).__init__(num_refine_stages, backbone, neck, bbox_head, frm_cfgs, refine_heads, train_cfg,
                                      test_cfg, pretrained, init_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_landms,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """Forward function."""
        # print('img_metas: ', img_metas)
        # gt_bboxes是list，list中每个tensor的size为11 5
        # gt_landms是list，list中每个tensor的size为11 6
        # gt_labels是list，list中每个tensor的size为11
        # print('gt_bboxes: ', gt_bboxes)
        # print('gt_landms: ', gt_landms)
        # print('gt_labels: ', gt_labels)
        losses = dict()
        x = self.extract_feat(img)
        
        # outs是list len为5 表示5个尺度的预测值
        # cls 的预测tensor shape为1 90 128 120
        # bbox 的预测tensor shape为1 45 128 120
        # landm 的预测tensor shape为1 54 128 120
        outs = self.bbox_head(x)
        # print('outs0: ', outs[0][0].shape)
        # print('outs1: ', outs[1][0].shape)
        # print('outs2: ', outs[2][0].shape)

        loss_inputs = outs + (gt_bboxes, gt_landms, gt_labels, img_metas)
        # 第一阶段不预测关键点，只在refine阶段预测
        # loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        # print('len(loss_inputs): ', len(loss_inputs))
        loss_base = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses[f's0.{name}'] = value
        
        # print('losses-bbox: ', losses)
        # f = open('/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-101-landmfliter-test/data.txt', 'a+')
        # f.write('imgmetas-' + str(img_metas) + '\n')
        # f = open('/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-101-landmfliter-test/data-bbox.txt', 'a+')
        # f.write('imgmetas-' + str(img_metas) + '\n')
        # f = open('/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-101-landmfliter-test/data-1.txt', 'a+')
        # f.write('imgmetas-' + str(img_metas) + '\n')
        
        # rois = self.bbox_head.filter_bboxes_landm(*outs)
        rois = self.bbox_head.filter_bboxes(*outs)
        # # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            lw = self.train_cfg.stage_loss_weights[i]

            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            # print('refine_outs0: ', outs[0][0].shape)
            # print('refine_outs1: ', outs[1][0].shape)
            # print('refine_outs2: ', outs[2][0].shape)
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_landms, img_metas)
            # print('loss_inputs: ', len(loss_inputs))
            loss_refine = self.refine_head[i].loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
            for name, value in loss_refine.items():
                losses[f'sr{i}.{name}'] = ([v * lw for v in value]
                                           if 'loss' in name else value)

            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

        # print('losses-refine: ', losses)

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        # print('test-test')
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # print('outs: ', len(outs))
        # print('outs: ', len(outs[0]))
        # print('outs: ', outs[2][0].shape)

        rois = self.bbox_head.filter_bboxes(*outs)
        # 只用关键点来做
        # rois = self.bbox_head.filter_bboxes_landm(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

        # # print('routs: ', len(outs))
        # # print('routs: ', len(outs[0]))
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        res_list = self.refine_head[-1].get_bboxes(*bbox_inputs, rois=rois)
        # 只用第一阶段不要refine阶段
        # res_list = self.bbox_head.get_bboxes(*bbox_inputs)
        # print('res_list: ', res_list)
        # print('det_res: ', det_res)

        bbox_results = [
            rbboxlandm2result(det_results, det_labels, self.refine_head[-1].num_classes)
            for det_results, det_labels in res_list
        ]
        # 返回bbox结果和landms结果
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    **kwargs):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (torch.Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_landm_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]
        else:
            bbox_landm_result, segm_result = result, None
        bboxes_landms = np.vstack(bbox_landm_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_landm_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_rbboxes_landms(
            img,
            bboxes_landms,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img
